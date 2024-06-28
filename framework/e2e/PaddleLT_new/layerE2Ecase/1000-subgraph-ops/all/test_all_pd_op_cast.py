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
    class PrimitiveOp_b424a347d9b0849cec6584fe74e3075f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_bba4ce69d03b2589504a7449fa145012(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f098e1c95c87764c32c49b9c17ee070(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba4ce69d03b2589504a7449fa145012
        def get_inputs(self):
            return [
                paddle.to_tensor([300.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0ad0eafa5c3e79d8e997530d47a029f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(3549, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_7915592b935e4847aaf184d0b09e7b57(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    
    class PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b71beb67d8235206c5d5a08dc503a143(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_bb1b6a54dac222638d571b3635bb7699(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb94b28ee4f98c79d4c881970a09b471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_6a7efc84ae558a9bf32aa0fcedd3ad9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9e89e028c6d753fa4b1735f1872f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_32a19618a8f19392b62d9a801d8cac68(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f89d72b0790c24d0db50e2580374a497(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32a19618a8f19392b62d9a801d8cac68
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_6a3f0452564d385d91569fde79419392(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba4ce69d03b2589504a7449fa145012
        def get_inputs(self):
            return [
                paddle.to_tensor([100.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_9ade39e9117dc96145d7ae1006aab434(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_67046a6efbf3a7a1bd2ad9436cd0f103(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_67046a6efbf3a7a1bd2ad9436cd0f103(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_750f2064f989d5e1d0cc83ff1b96d129(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2100], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7488d14442097c3895d45277f5c644b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_750f2064f989d5e1d0cc83ff1b96d129
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            ]


    
    class PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_788a064f7b2d556d556a560d91722178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([128], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6a444441765796f1c39217f0ee921af5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([16], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4c5ee52e144167a1efe8c05d86d3f96e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_417d45b869acecca151b288c7c758eef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[96], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_91134331b5f3ef6c01b0989c8ce1a1f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_417d45b869acecca151b288c7c758eef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[96], dtype='int64'),
            ]


    
    class PrimitiveOp_5e40b041dd714cf501c1b82e1156c262(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[48], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b877f488bd9bf193577eff551379ea6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e40b041dd714cf501c1b82e1156c262
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[48], dtype='int64'),
            ]


    
    class PrimitiveOp_cb138e8ea9ebc6a55103396e5d894c2f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2917cc5d772c91c481b11f43a32cb690(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb138e8ea9ebc6a55103396e5d894c2f
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], dtype='int64').reshape([24]),
            ]


    
    class PrimitiveOp_b2478cc76adbd9d449b1a74b60c90c34(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[12096, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc96071e7b13785a6386a685b462fafa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2478cc76adbd9d449b1a74b60c90c34
        def get_inputs(self):
            return [
                paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc96071e7b13785a6386a685b462fafa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2478cc76adbd9d449b1a74b60c90c34
        def get_inputs(self):
            return [
                paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_9ac4ca313e7fbd3c10e159cd7bd7c0be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f763c3b59876e2220b915dea7b16ad9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ac4ca313e7fbd3c10e159cd7bd7c0be
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_2694f0c7f667b1479a04c20f6961e367(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    
    class PrimitiveOp_d69ec1bcb65db19225d44841159e54d2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b31995b8aedb34b8ce56a4ddfbbba328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d69ec1bcb65db19225d44841159e54d2
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_60122ffd4eb2fc5ee94a65f12f7cf2e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_60122ffd4eb2fc5ee94a65f12f7cf2e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_895e28dbe52d7a21dec773cd0e0db85a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f73b1c9686b81e23bad03d63d4117473(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895e28dbe52d7a21dec773cd0e0db85a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4765012264251709, 0.2170112580060959, 0.4702689051628113, 0.4369574785232544, 0.244343563914299, 0.49530693888664246, 0.3121092915534973, 0.0963035523891449, 0.38352274894714355, 0.009155333042144775, 0.4616371691226959, 0.020641636103391647, 0.15587159991264343, 0.02491425909101963, 0.443154513835907, 0.24451406300067902], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_55f22f83b00b1b428aed96fa4c79ec74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_bb5d9aa93516f85a282dce087895dccf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_d0a5d04052470b0d8d6a1caca8c64254(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(7581, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_802fe6cf4c2dbdee0f05c2b952759981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c9681b531783052c0e4d1094737f42e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_dece7d522d618b4505b77810c7a853a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4725, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d70e46bb12ce27d57cd853881bd577cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32a19618a8f19392b62d9a801d8cac68
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_0153e4557c71d082b53718bee695fa03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(577, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ef1ccc76a8035835ee9d8a5de828c7d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_87c9b2cdd6d12ef417ebde9610eb701e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_32904982dc10676823c0da88bc0a8ed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_a96bb893b3bee2c4f3cc385db15ee23f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 4], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_093bbe1c32bc4af6b766053e4b36451f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a96bb893b3bee2c4f3cc385db15ee23f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 4], dtype='int32'),
            ]


    
    class PrimitiveOp_e9f75f925563c8e158a4bbd9df0cbd86(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d84dcba521165ef0567366184e87f278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9f75f925563c8e158a4bbd9df0cbd86
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_c8e9706fd8b632c9ae21ecd544ccfacf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 68], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_051b130448bd6f1cac9102c5fbefcd77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8e9706fd8b632c9ae21ecd544ccfacf
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 68], dtype='int32'),
            ]


    
    class PrimitiveOp_d5dc131155abc10eb38db2181b367f25(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cb2405de9d79062fc6c1bc4f9559da75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5dc131155abc10eb38db2181b367f25
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a97867b7d56f67fe14cfbb40cd5e1c96(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_38daa3e7dcdc75944b2daaa6aa79c5a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97867b7d56f67fe14cfbb40cd5e1c96
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1723, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_6fe7aa1a1a3cb0946799831807fa632e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0186bf9c3fe25f025061454de6bc085b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8400, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_0fafe3974c19dea491ba88de1fd22485(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[64], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_57a077c75e76b2ca8a7d7aab0cc719f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fafe3974c19dea491ba88de1fd22485
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    
    class PrimitiveOp_d919a13f509de0d72f32a49750393c62(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[32], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1e166390a59daa6df6f05aec6a7c6c93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d919a13f509de0d72f32a49750393c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    
    class PrimitiveOp_9b41322013e40feea1fdc5a0eb2854c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f5fb66a94dfd8a9e445bc1b28f3820b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b41322013e40feea1fdc5a0eb2854c4
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
            ]


    
    class PrimitiveOp_5004ac6c4b720e53d3370381ff2a8e64(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5376, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6607080a9771f9242ca2d296e5f8335e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5004ac6c4b720e53d3370381ff2a8e64
        def get_inputs(self):
            return [
                paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6607080a9771f9242ca2d296e5f8335e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5004ac6c4b720e53d3370381ff2a8e64
        def get_inputs(self):
            return [
                paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0ad0eafa5c3e79d8e997530d47a029f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(3549, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_55d584f3671d10c8df4d8cbab851020f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_012a6fae748b97b18b09071037379250(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d9b56317e8be88e0b14c6e99c242dea0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d146990ffb593c3ee23bc0a92d4e7488(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9b90e8b9dcb25890df9fe05cfdddf060(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a96bb893b3bee2c4f3cc385db15ee23f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_b5808b9f97adb10980262fd45824f424(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9f75f925563c8e158a4bbd9df0cbd86
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_2bb877d9ef9b43e9c5cb355f232d66ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8e9706fd8b632c9ae21ecd544ccfacf
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_1d5c8b71841febf243d84964f258a665(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5dc131155abc10eb38db2181b367f25
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_716c4a625c7d545a4d83c99378af6d3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97867b7d56f67fe14cfbb40cd5e1c96
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5498, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_a82e72e1a242fa370d52a16a5ebfb6b9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_26626a2dbf2d8ae4805058a313b2a58a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a82e72e1a242fa370d52a16a5ebfb6b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cfd60f8182e829e651f8ec58183f478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bcf3aa8f2195df151329a6e16ce74754(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f675092cea2b25bd1f22e00a7a5aa1fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_964f9e01a90bc3e437cdb7b321fcc0c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(98, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_22fe7e9a82cc0ad3ed4ab7ca6e6fdc93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(99, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_dba0fe45cf9a305cc7dbcc5b36c5d148(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895e28dbe52d7a21dec773cd0e0db85a
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3fa446a52be134607b410e6bfa9b7946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_3fa446a52be134607b410e6bfa9b7946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2156318ac36bb8e1ee334d298d3f2bff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(192, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_ab3e73b499fdf79d13ffe3fc8fdab3c6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d072a89a02e192c990e7686c7fd59ab1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab3e73b499fdf79d13ffe3fc8fdab3c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1fe1374ad3830594db3447855a8e2567(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 192, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c9681b531783052c0e4d1094737f42e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_92035424d16071704640fceac9cb7ba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d69ec1bcb65db19225d44841159e54d2
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_802fe6cf4c2dbdee0f05c2b952759981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c9681b531783052c0e4d1094737f42e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_012a6fae748b97b18b09071037379250(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9b56317e8be88e0b14c6e99c242dea0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c9681b531783052c0e4d1094737f42e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bcf3aa8f2195df151329a6e16ce74754(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_f51da3aa8ea72907a3b1130e603363d0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d1afde021d493e1bfb21897979dab25d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f51da3aa8ea72907a3b1130e603363d0
        def get_inputs(self):
            return [
                paddle.to_tensor([False, True, False, False, False, False], dtype='bool').reshape([6]),
            ]


    class TestPrimitiveOp_ad7ab7da8ed2ead67d8fde22b8a0a257(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f51da3aa8ea72907a3b1130e603363d0
        def get_inputs(self):
            return [
                paddle.to_tensor([False, False, False, False, False, False], dtype='bool').reshape([6]),
            ]


    class TestPrimitiveOp_32904982dc10676823c0da88bc0a8ed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_093bbe1c32bc4af6b766053e4b36451f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a96bb893b3bee2c4f3cc385db15ee23f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_d84dcba521165ef0567366184e87f278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9f75f925563c8e158a4bbd9df0cbd86
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_6ba14d4f6dee5705089eb29cf7c67145(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 76], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9a4e52be0456619c692de9c5514e7f13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ba14d4f6dee5705089eb29cf7c67145
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 76], dtype='int32'),
            ]


    class TestPrimitiveOp_9268e868d706b4f8cc434c5b6eb5d8fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5dc131155abc10eb38db2181b367f25
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d530e901a0cd8397f724d9fa6ed83c35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97867b7d56f67fe14cfbb40cd5e1c96
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1759, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_fd6e696ccb777cd0431be6ca1519bcad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_80c5526749a98c18bd89e09717569b90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd6e696ccb777cd0431be6ca1519bcad
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a06f609b47a9fb3b8019bb20a43feeef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_255cc98a9269f9f18d3720e27ce2da7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd6e696ccb777cd0431be6ca1519bcad
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a06f609b47a9fb3b8019bb20a43feeef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_802fe6cf4c2dbdee0f05c2b952759981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0b0edc8276c10fc9d7bb0e027ab42527(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(28, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_de07304dbd124756772af50e24fac360(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(50, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4ef536d038f8c4fbedd7c2cb6605bec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_182239e1ffe3ed9bf865d9aa1f21632e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4116, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2d92b41333621cbb8087ed27db437c55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b92ead0ec046724cf415e9328b0a53f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_d839ae06b2506fad2dd389aee3e06599(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[80], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d31778eed4a7f24a47433b1207464fdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d839ae06b2506fad2dd389aee3e06599
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[80], dtype='int64'),
            ]


    
    class PrimitiveOp_84fd698027ded646cca2abe3206bc5df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68af7152486f41e6e844964237fedec5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84fd698027ded646cca2abe3206bc5df
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
            ]


    
    class PrimitiveOp_edef3039412121f63e84a43ce2c79182(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3bdfa9ffadc1c239c606848250aea563(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edef3039412121f63e84a43ce2c79182
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype='int64').reshape([20]),
            ]


    
    class PrimitiveOp_bed2851ee43df48dcf93054fbbcf5e10(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8400, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c14fd6497d9f17b9f0f2c6c3dd3d42f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bed2851ee43df48dcf93054fbbcf5e10
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c14fd6497d9f17b9f0f2c6c3dd3d42f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bed2851ee43df48dcf93054fbbcf5e10
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4dd417b86875586a21858d0a4d4d6810(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e75230232bf81471cad7146039bfab86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dd417b86875586a21858d0a4d4d6810
        def get_inputs(self):
            return [
                paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7b468f77da28e22ebb426950d328acfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895e28dbe52d7a21dec773cd0e0db85a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3306543529033661, 0.3834773600101471, 0.49611005187034607, 0.11812765151262283, 0.478644460439682, 0.32757896184921265, 0.3514584004878998, 0.3204523026943207, 0.48325031995773315, 0.25242266058921814, 0.4594441056251526, 0.10603760182857513, 0.09884601831436157, 0.039876293390989304, 0.12443646788597107, 0.20525671541690826, 0.031096970662474632, 0.08158884942531586, 0.2653093934059143, 0.3785844147205353, 0.4728078544139862, 0.47888636589050293, 0.37857428193092346, 0.030704200267791748], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_773fd4d7fce0f317f5804c52e34c595f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
            ]


    class TestPrimitiveOp_92bca5d268bb244f177c1c068a51dcc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
            ]


    
    class PrimitiveOp_3f4d0e63d97c505db8fff9e1194f2eb5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d54f5547e96de046763500ad795bcaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f4d0e63d97c505db8fff9e1194f2eb5
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6939955949783325], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_b71beb67d8235206c5d5a08dc503a143(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_bb1b6a54dac222638d571b3635bb7699(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb94b28ee4f98c79d4c881970a09b471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_6a7efc84ae558a9bf32aa0fcedd3ad9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9e89e028c6d753fa4b1735f1872f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c92e0ec02de5625fd55b34449d684532(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6069, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_25805ace3e67d7710cb359976372fb7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c308c5638e72ac46b211a16e9a119600(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a96bb893b3bee2c4f3cc385db15ee23f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_011efcfc1a0293f5b388f5ffaf3147e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9f75f925563c8e158a4bbd9df0cbd86
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6630fcad613198455156351ca14c069e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8e9706fd8b632c9ae21ecd544ccfacf
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_26b4a2b216a64b713da47cfee91f0328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5dc131155abc10eb38db2181b367f25
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a029daaf64c75d80157fc774a65b4df2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97867b7d56f67fe14cfbb40cd5e1c96
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1538, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_32904982dc10676823c0da88bc0a8ed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_32904982dc10676823c0da88bc0a8ed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_1827f5822e45a11bd5e670a4b29ad66f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3549], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_23bda630dca4976d4495720f335e4e8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1827f5822e45a11bd5e670a4b29ad66f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    class TestPrimitiveOp_3543022539aa86acc1acc8c5cc6a59a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dd417b86875586a21858d0a4d4d6810
        def get_inputs(self):
            return [
                paddle.to_tensor([8, 2], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_77d8e041bbee4adad3a29fc4bea8b00f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895e28dbe52d7a21dec773cd0e0db85a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.004606406204402447, 0.23123939335346222, 0.4518365263938904, 0.139329195022583], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_ed70912a87c6e9b766ec49dc5a1bced8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_61a5b77fcd93a97e5f2607de33269744(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_904e9b9355d7e13194363e52c0f2f695(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(52, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b5489bc08cb69df0cd70fb6389691632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(202, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_bb8d3995b3b5a551c656ee8b23ee0c42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1025, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_6472103106d2b18ba79b111b44349ece(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[14], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_08a93a65b2fdf48cb93dcd473854c5a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6472103106d2b18ba79b111b44349ece
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype='int64').reshape([14]),
            ]


    
    class PrimitiveOp_ecc23effe974611db28e250e4c3e5354(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[14, 14, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2e739c1531583ca947b98d424070081f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecc23effe974611db28e250e4c3e5354
        def get_inputs(self):
            return [
                paddle.uniform([14, 14, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c527f1a745b53c2f00135a1a3e45cd64(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[14, 14, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3577090808847984eb3185a3b1a2c675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c527f1a745b53c2f00135a1a3e45cd64
        def get_inputs(self):
            return [
                paddle.uniform([14, 14, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2b6bb2ed6f5ff2db0d3b174c6413b7cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[28], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c15075ac2b0f4dc3047a15b0345dfe00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b6bb2ed6f5ff2db0d3b174c6413b7cb
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], dtype='int64').reshape([28]),
            ]


    
    class PrimitiveOp_86a5a8a79c6c5b9e03d5b869e5e8a103(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[28, 28, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff7406548ba643fb37280e55696548a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86a5a8a79c6c5b9e03d5b869e5e8a103
        def get_inputs(self):
            return [
                paddle.uniform([28, 28, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a2214949df8e1cf8049f839e7a30e80b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[28, 28, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_afb7f5442fbfbc5be5bce5f1d0cc1943(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a2214949df8e1cf8049f839e7a30e80b
        def get_inputs(self):
            return [
                paddle.uniform([28, 28, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_de7c140b7de9b0938673704688fbbb95(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d798a4a46ab85834fa927ba694fd9d98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de7c140b7de9b0938673704688fbbb95
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[56], dtype='int64'),
            ]


    
    class PrimitiveOp_ce1a67b3b1ff3815092f9cb98d9b8434(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56, 56, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c827c6eb665d82c004f8be91bc26ed6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce1a67b3b1ff3815092f9cb98d9b8434
        def get_inputs(self):
            return [
                paddle.uniform([56, 56, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_13e797d8acd75d28933cfd7c6ce383fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56, 56, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_96ffc4db00ce359c0afed1c8469bbbcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13e797d8acd75d28933cfd7c6ce383fc
        def get_inputs(self):
            return [
                paddle.uniform([56, 56, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_60122ffd4eb2fc5ee94a65f12f7cf2e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_60122ffd4eb2fc5ee94a65f12f7cf2e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_de5243416defd095a8f4901acb914079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(104, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_de5243416defd095a8f4901acb914079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(104, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_069aabc70ea7b88a2560e3bc2d503ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_069aabc70ea7b88a2560e3bc2d503ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_fe221e49a6d13fbc49b482014be7a695(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4116], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eb5a2da78216763e318a6663e162481b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe221e49a6d13fbc49b482014be7a695
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_e84ab84058793cee2719f0ee499707f4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_10cdb2ae7d8bba2fdab34a87674d6a63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e84ab84058793cee2719f0ee499707f4
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5be6741a1a32bd81e70e6152de3c2ebf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e84ab84058793cee2719f0ee499707f4
        def get_inputs(self):
            return [
                paddle.to_tensor(7, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8f098e1c95c87764c32c49b9c17ee070(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba4ce69d03b2589504a7449fa145012
        def get_inputs(self):
            return [
                paddle.to_tensor([300.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cec54815ac05282cf7f41838d06b6482(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f4ef536d038f8c4fbedd7c2cb6605bec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_069aabc70ea7b88a2560e3bc2d503ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c2b6cda6f44255603efac7e661198b42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a96bb893b3bee2c4f3cc385db15ee23f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_f55584b4c8fe7b3654c44019f84148c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9f75f925563c8e158a4bbd9df0cbd86
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a116b9b918ad3cafb4b9e7d42ed902f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8e9706fd8b632c9ae21ecd544ccfacf
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_b7ed92cd4ddf4a97bb8b7532986e77cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5dc131155abc10eb38db2181b367f25
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1dce24081bd7a17a1bcae452a0adad9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97867b7d56f67fe14cfbb40cd5e1c96
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2135, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_1f617b341f517a443d6c09a453fc7f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(14, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8d063c09f60fcc9258d34e3e9b863cd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(25, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f8d26dc573bfd532cf6be4dab502c17c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_bcc416c8b660a6ebbce01826833cec60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a96bb893b3bee2c4f3cc385db15ee23f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_007406bdde70a344c5141d24a593577c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9f75f925563c8e158a4bbd9df0cbd86
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f31c28879a61f2888f5fb9b2c83447ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8e9706fd8b632c9ae21ecd544ccfacf
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_9dae91e431b15e35fd8ebcc41c4757ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5dc131155abc10eb38db2181b367f25
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e0fe7c07a42d0a959d8888680294131(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97867b7d56f67fe14cfbb40cd5e1c96
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4590, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_8a2c86d3a221afc1c9dfefd13e298794(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32a19618a8f19392b62d9a801d8cac68
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[6, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_4b1d2e5405b199bdf798aad70af5c9ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ac4ca313e7fbd3c10e159cd7bd7c0be
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_67046a6efbf3a7a1bd2ad9436cd0f103(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c84e79d2f834b1b3e6bb8ad944612881(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a96bb893b3bee2c4f3cc385db15ee23f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_f45973fbea6e3a8d3955b6aa0962d90f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9f75f925563c8e158a4bbd9df0cbd86
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_cd6d79ccd9d05602e79c2a08e3d58784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8e9706fd8b632c9ae21ecd544ccfacf
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_688d9a05d975987d5fb3d4e5e537eb97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5dc131155abc10eb38db2181b367f25
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d2a18ae85b15458035f65e25e971d44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97867b7d56f67fe14cfbb40cd5e1c96
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1042, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_159853c5be6e7ff71d2a2861bf97f1ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(9261, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bcf3aa8f2195df151329a6e16ce74754(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_e7d9a4859786add76a55da68a87b5f53(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[68], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0ad58846f902324f75475abf194f1553(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e7d9a4859786add76a55da68a87b5f53
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[68], dtype='int64'),
            ]


    
    class PrimitiveOp_ade11628c930b53312f35cea2b97b307(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[34], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d0fb47adbe2e30f6e74c529643e5519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ade11628c930b53312f35cea2b97b307
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[34], dtype='int64'),
            ]


    
    class PrimitiveOp_6773edf339f76fbc72008ad854b1bd67(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[17], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ce018a44e36dfda64be1cd920c8c296(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6773edf339f76fbc72008ad854b1bd67
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype='int64').reshape([17]),
            ]


    
    class PrimitiveOp_c1dbebddafbd7d04dc39bd6f57b1cf78(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6069, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5e43b2fad1f08f34b6e0ff928674cd68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1dbebddafbd7d04dc39bd6f57b1cf78
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e43b2fad1f08f34b6e0ff928674cd68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1dbebddafbd7d04dc39bd6f57b1cf78
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_26626a2dbf2d8ae4805058a313b2a58a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a82e72e1a242fa370d52a16a5ebfb6b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cfd60f8182e829e651f8ec58183f478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_26626a2dbf2d8ae4805058a313b2a58a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a82e72e1a242fa370d52a16a5ebfb6b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cfd60f8182e829e651f8ec58183f478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_26626a2dbf2d8ae4805058a313b2a58a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a82e72e1a242fa370d52a16a5ebfb6b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cfd60f8182e829e651f8ec58183f478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_eb531f5775a9f575efca9cb6a04c768b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(2048, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_85aeb4d4e07a00e5b71edc2c20b78bec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cced9358935e7d9dd08258b2af71a9d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85aeb4d4e07a00e5b71edc2c20b78bec
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66d2f2b01ac9cd13229fb16ed0c8f47e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2048, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_6a7efc84ae558a9bf32aa0fcedd3ad9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9e89e028c6d753fa4b1735f1872f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_bb1b6a54dac222638d571b3635bb7699(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb94b28ee4f98c79d4c881970a09b471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_b71beb67d8235206c5d5a08dc503a143(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_617dab664ac3ba7b80a678548a4279e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_617dab664ac3ba7b80a678548a4279e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_ffd12346278a3c3421b711fdcabd80ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([16, 16, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2d1f7851632fad7b8414226f4a5d21ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([8]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2d1f7851632fad7b8414226f4a5d21ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([8]),
            ]


    class TestPrimitiveOp_ed262ab14aa891a8cda5ddb3bdc37a79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([8, 8, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e244c47b82deaf9f39c116ef0e74df1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(2100, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5d856b0d10d31e2fbab897d9ed483f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a82e72e1a242fa370d52a16a5ebfb6b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cfd60f8182e829e651f8ec58183f478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5d856b0d10d31e2fbab897d9ed483f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a82e72e1a242fa370d52a16a5ebfb6b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cfd60f8182e829e651f8ec58183f478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5d856b0d10d31e2fbab897d9ed483f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a82e72e1a242fa370d52a16a5ebfb6b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cfd60f8182e829e651f8ec58183f478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_eb531f5775a9f575efca9cb6a04c768b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(2048, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_36b451af27f4d315b0fb6e58d16beaf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85aeb4d4e07a00e5b71edc2c20b78bec
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66d2f2b01ac9cd13229fb16ed0c8f47e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2048, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8d3995b3b5a551c656ee8b23ee0c42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1025, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b51694e6ab84e392d6f6ac9902de4f15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8092276b18087f3b9bc29bdfc1e076(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a96bb893b3bee2c4f3cc385db15ee23f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_b3bbd14996ac32b13fdd24c5fde27fcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9f75f925563c8e158a4bbd9df0cbd86
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9de10a345f49f298d9b46f35380c9471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8e9706fd8b632c9ae21ecd544ccfacf
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_d1f1220f64c3eea532143cba2d954af3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5dc131155abc10eb38db2181b367f25
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_333355b96e5dc9e50349f24318328b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97867b7d56f67fe14cfbb40cd5e1c96
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2339, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_802fe6cf4c2dbdee0f05c2b952759981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ec485f2c5df474efa49cb9604a0c6325(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_56fbf303f5ba115976bb6808b2adae21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a96bb893b3bee2c4f3cc385db15ee23f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_fe37ecd883e14d7f0f1a749bd9d7bacd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9f75f925563c8e158a4bbd9df0cbd86
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7f0f9aa634acb8c7b728582c50bc3dd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8e9706fd8b632c9ae21ecd544ccfacf
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_16f2d7b907dbb0ee84d5a0a6f076aa3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5dc131155abc10eb38db2181b367f25
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae269cc888f39d0b34a86e2d1d56412e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97867b7d56f67fe14cfbb40cd5e1c96
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3063, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_980c04af2955f51824b4abca7a6deb38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7cf166e35d817e3cbad4ce5887c8d32c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a96bb893b3bee2c4f3cc385db15ee23f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_e81f9a554a090ae4fe68e27ea5e0233f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9f75f925563c8e158a4bbd9df0cbd86
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b7d60db1ea8301a6219686e14d5e692d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8e9706fd8b632c9ae21ecd544ccfacf
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_8bb08a2e889f105b101c5cfb395f0bf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5dc131155abc10eb38db2181b367f25
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc704e07f595298fdde5424674da0e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97867b7d56f67fe14cfbb40cd5e1c96
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3822, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6a3f0452564d385d91569fde79419392(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba4ce69d03b2589504a7449fa145012
        def get_inputs(self):
            return [
                paddle.to_tensor([100.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_cca72d2a6439ce40b2a2d3b7b6c0a4d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11109, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f65445d4f28e8755b69d5275765232b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_63fb5f540d40add592fdf67694c9eea9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32a19618a8f19392b62d9a801d8cac68
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_bfa8aaf3ceda90e4b60cab96d0758ad9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_edbbe79e5dfef890a0a02ea2891d500b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([11], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e9bb4cc7be02a14ac6eea0421d4d2463(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([384], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bdf72173d9ed3edb7112612db5c3d998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([28], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f51299282ffee062040fa3d2b67aeb38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([77], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_bc9232f8ccf5abe5be92c2749fdc3a6c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[152], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fc1175cfddfcd5e5bfd8590d9f14a0be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc9232f8ccf5abe5be92c2749fdc3a6c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[152], dtype='int64'),
            ]


    
    class PrimitiveOp_5ed02a1c76d46a1e5630efd7b9502950(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6878171d55b6bba9cc9f848db81eac08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ed02a1c76d46a1e5630efd7b9502950
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[100], dtype='int64'),
            ]


    
    class PrimitiveOp_27430bd5e8a381a6c05a561dae8575cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 152, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6f5db86631ae22d24a550eadb1e4f5b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27430bd5e8a381a6c05a561dae8575cf
        def get_inputs(self):
            return [
                paddle.uniform([100, 152, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_461554130b035570edf5be277ff9cfd8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 152, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ebd95ae730efae30290b72b71ead83a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461554130b035570edf5be277ff9cfd8
        def get_inputs(self):
            return [
                paddle.uniform([100, 152, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c4cd883e14789757030af4095b78018b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[76], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_98ca641887034778e0d3e89b44a003c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4cd883e14789757030af4095b78018b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[76], dtype='int64'),
            ]


    
    class PrimitiveOp_a973a591c8ee767df009b149c504545d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[50], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_765ea9296bf58e935e728be4a49c723a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a973a591c8ee767df009b149c504545d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[50], dtype='int64'),
            ]


    
    class PrimitiveOp_76c708e82f7438fe00d117e99cb675aa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[50, 76, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_33433d7192c0af9df49c49da8351a45e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76c708e82f7438fe00d117e99cb675aa
        def get_inputs(self):
            return [
                paddle.uniform([50, 76, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4795b870d1c1d9292ce5a5bd32164fff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[50, 76, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6070d6412a2bee9e49cab59949d1642e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4795b870d1c1d9292ce5a5bd32164fff
        def get_inputs(self):
            return [
                paddle.uniform([50, 76, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_37f23af4254b3e668dbcb1353c7a9ab2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[38], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c1207e13bfdcda2dac72c83cc2d171fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37f23af4254b3e668dbcb1353c7a9ab2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[38], dtype='int64'),
            ]


    
    class PrimitiveOp_7b62c0a086d1a37da5b737fa9a21df82(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[25], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2ca51e31b1e75a5e1dc09603e8d95188(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b62c0a086d1a37da5b737fa9a21df82
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype='int64').reshape([25]),
            ]


    
    class PrimitiveOp_bb103be57129aa533357a57f599d0847(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[25, 38, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7671b3728a440a899bcac3b7f102a19a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb103be57129aa533357a57f599d0847
        def get_inputs(self):
            return [
                paddle.uniform([25, 38, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_81fb568efc0137a67b6dcf28fb6fd715(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[25, 38, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_989e0aa9a98729509c80c4589329c6d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81fb568efc0137a67b6dcf28fb6fd715
        def get_inputs(self):
            return [
                paddle.uniform([25, 38, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e51fc23940a6b12d25b5badc3dd697fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[19], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2a1e1375f66f2801ae2d952bb3f17d96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e51fc23940a6b12d25b5badc3dd697fb
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype='int64').reshape([19]),
            ]


    
    class PrimitiveOp_d8bf18062474d6c43ae826be76808706(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[13], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_67d24f4eb63084622ce12444c999a9f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8bf18062474d6c43ae826be76808706
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype='int64').reshape([13]),
            ]


    
    class PrimitiveOp_05b29b433ab0937e8fe16428b44dba47(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[13, 19, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f85b71fe4b3c9e490193fe8c05887450(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05b29b433ab0937e8fe16428b44dba47
        def get_inputs(self):
            return [
                paddle.uniform([13, 19, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_95022d4d291260aa4d534f4364e777c5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[13, 19, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aee9d3190043e7fc7d92fbdcd7593de6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95022d4d291260aa4d534f4364e777c5
        def get_inputs(self):
            return [
                paddle.uniform([13, 19, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eaa4743693b05cf99345cfaa026a990b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ca3214e01eb4e38c3d02a439039c6364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eaa4743693b05cf99345cfaa026a990b
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64').reshape([10]),
            ]


    
    class PrimitiveOp_543120e7a9a61eff3ed407a860346bec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a8afb99ac3c5c6ec74a4daa1e03e5d2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_543120e7a9a61eff3ed407a860346bec
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6], dtype='int64').reshape([7]),
            ]


    
    class PrimitiveOp_62437164d4281759d954b859874051ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7, 10, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_186427a771edd8be4562186d8ceeb782(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62437164d4281759d954b859874051ee
        def get_inputs(self):
            return [
                paddle.uniform([7, 10, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e35b7f1a5582b4528f9ed2c763190bb2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7, 10, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_175f7cfaba243fb79d6e39220e742e58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e35b7f1a5582b4528f9ed2c763190bb2
        def get_inputs(self):
            return [
                paddle.uniform([7, 10, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_d54b63e48565a15ff1c7840305b4bd56(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0ef0b1ede8ea0900d98b87ad023f0b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d54b63e48565a15ff1c7840305b4bd56
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b92ead0ec046724cf415e9328b0a53f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6bebc880fd1836884b6ff2ed27f2c1e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd6e696ccb777cd0431be6ca1519bcad
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a06f609b47a9fb3b8019bb20a43feeef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c9681b531783052c0e4d1094737f42e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d1ba26cfb1b248a545fe9250efbb97e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895e28dbe52d7a21dec773cd0e0db85a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.34141385555267334, 0.08070738613605499, 0.03860168904066086, 0.49608147144317627, 0.41707998514175415, 0.07405710965394974, 0.39049777388572693, 0.14312376081943512, 0.4575801193714142, 0.44072431325912476, 0.48256993293762207, 0.10408809781074524, 0.3957173824310303, 0.045056845992803574, 0.2210846096277237, 0.11286043375730515, 0.109720878303051, 0.17332060635089874, 0.47122129797935486, 0.48317795991897583], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_91c1e9c4d62c453dff72e86b2b60229f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_d0aea1398949b006b5e02c7358179872(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_b71beb67d8235206c5d5a08dc503a143(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_bb1b6a54dac222638d571b3635bb7699(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb94b28ee4f98c79d4c881970a09b471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_6a7efc84ae558a9bf32aa0fcedd3ad9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9e89e028c6d753fa4b1735f1872f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_cec54815ac05282cf7f41838d06b6482(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_069aabc70ea7b88a2560e3bc2d503ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c2b6cda6f44255603efac7e661198b42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a96bb893b3bee2c4f3cc385db15ee23f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_f55584b4c8fe7b3654c44019f84148c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9f75f925563c8e158a4bbd9df0cbd86
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a116b9b918ad3cafb4b9e7d42ed902f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8e9706fd8b632c9ae21ecd544ccfacf
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_eac23069248c9df8ab603ad006bb9e7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5dc131155abc10eb38db2181b367f25
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5452b1e5a8ac100fdb09aea1878bae3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97867b7d56f67fe14cfbb40cd5e1c96
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2057, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_856ed2ef3283730c2d173a837b3d519f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a82e72e1a242fa370d52a16a5ebfb6b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cfd60f8182e829e651f8ec58183f478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0af5f14fbb5c825ceafed43ed9515b50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(3024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ef1ccc76a8035835ee9d8a5de828c7d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_4c1b262b9184613bdc87720c8f1ff057(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[72], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_95bd444b57f07058e3fb7d2baece734b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c1b262b9184613bdc87720c8f1ff057
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[72], dtype='int64'),
            ]


    
    class PrimitiveOp_6d9859c8b7984fe1f96fcf63c2094932(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[36], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1a948e78f582b70a2b626d3c8dc11e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d9859c8b7984fe1f96fcf63c2094932
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    
    class PrimitiveOp_c5f14d69836727d2e3c01e8580db0dd6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[18], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8420ccb25952117d49fdc7f4a3f2277(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5f14d69836727d2e3c01e8580db0dd6
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype='int64').reshape([18]),
            ]


    
    class PrimitiveOp_6514caaea4dfe18793e38eeb8c30609d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6804, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_495f694623b01c30c75fb2c8f4f790cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6514caaea4dfe18793e38eeb8c30609d
        def get_inputs(self):
            return [
                paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_495f694623b01c30c75fb2c8f4f790cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6514caaea4dfe18793e38eeb8c30609d
        def get_inputs(self):
            return [
                paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_ea0df210b00f923541bf6bcc99305f0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1174, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e8df93d5e317d77438ed6c60e8173dbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d69ec1bcb65db19225d44841159e54d2
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_730322fdc76616c6ac52fcabfe2d2bf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5d856b0d10d31e2fbab897d9ed483f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a82e72e1a242fa370d52a16a5ebfb6b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cfd60f8182e829e651f8ec58183f478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_cec54815ac05282cf7f41838d06b6482(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2d92b41333621cbb8087ed27db437c55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b92ead0ec046724cf415e9328b0a53f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b8bd8f4fdc409fe2bb83c605b010ff7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5d998325673a24e20286d59698f863bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8582e0f5d9f0ed17548a28e8414a7b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a96bb893b3bee2c4f3cc385db15ee23f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_d5f8286a254a9256713dd070aa7ea452(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9f75f925563c8e158a4bbd9df0cbd86
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5d26ad1965b1f325f543f2016bfe7eb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8e9706fd8b632c9ae21ecd544ccfacf
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_e7b6f9bfa2b4adc8688e1054e25c0f28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5dc131155abc10eb38db2181b367f25
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_038194afc32b5e0981fb11bba1131991(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97867b7d56f67fe14cfbb40cd5e1c96
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4189, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_ea0df210b00f923541bf6bcc99305f0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1174, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_b71beb67d8235206c5d5a08dc503a143(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_bb1b6a54dac222638d571b3635bb7699(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb94b28ee4f98c79d4c881970a09b471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_6a7efc84ae558a9bf32aa0fcedd3ad9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b0edd00e5009fdb5219a2bf8ee6b35
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9e89e028c6d753fa4b1735f1872f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8f098e1c95c87764c32c49b9c17ee070(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba4ce69d03b2589504a7449fa145012
        def get_inputs(self):
            return [
                paddle.to_tensor([300.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0ad0eafa5c3e79d8e997530d47a029f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(3549, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1e166390a59daa6df6f05aec6a7c6c93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d919a13f509de0d72f32a49750393c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1e166390a59daa6df6f05aec6a7c6c93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d919a13f509de0d72f32a49750393c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    
    class PrimitiveOp_f34095942dc59cfe8a61aaf9cc0a6a1f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[32, 32, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_81e10633f9089dd38f84624695712084(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f34095942dc59cfe8a61aaf9cc0a6a1f
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_57a077c75e76b2ca8a7d7aab0cc719f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fafe3974c19dea491ba88de1fd22485
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_57a077c75e76b2ca8a7d7aab0cc719f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fafe3974c19dea491ba88de1fd22485
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    
    class PrimitiveOp_ba3b6992b7cf4500fd957ef9235401df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[64, 64, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0fcdf590ec7a0638637cd0b5cb4f258c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba3b6992b7cf4500fd957ef9235401df
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb94b28ee4f98c79d4c881970a09b471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_90140664290a2ebea0c9ccc3b00815bb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[128], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_78615fc6d63cf353c3760780a2a9baa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90140664290a2ebea0c9ccc3b00815bb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_78615fc6d63cf353c3760780a2a9baa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90140664290a2ebea0c9ccc3b00815bb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    
    class PrimitiveOp_8272d93ef23a7780bc3b71c27884c666(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[128, 128, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b3a97447a766a3722ee7f4eb212d26e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8272d93ef23a7780bc3b71c27884c666
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9e89e028c6d753fa4b1735f1872f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_dbce5acac399f8d7da85a8ae5ec3c955(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 28, 28], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_497ed0e4711992e216a17b53ebac92d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbce5acac399f8d7da85a8ae5ec3c955
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_6a3f0452564d385d91569fde79419392(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba4ce69d03b2589504a7449fa145012
        def get_inputs(self):
            return [
                paddle.to_tensor([100.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_f7af1227d0f5a1e0abf18bee01d5e23e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d15a7cee783a544c3f85961383bec90a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7af1227d0f5a1e0abf18bee01d5e23e
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d15a7cee783a544c3f85961383bec90a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7af1227d0f5a1e0abf18bee01d5e23e
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_83e3fc79a4689b20c90c95dfce3c5178(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b68574bfc734e196aedabe6beac68f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83e3fc79a4689b20c90c95dfce3c5178
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            ]


    class TestPrimitiveOp_788a064f7b2d556d556a560d91722178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([128], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6a444441765796f1c39217f0ee921af5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([16], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4c5ee52e144167a1efe8c05d86d3f96e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_91134331b5f3ef6c01b0989c8ce1a1f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_417d45b869acecca151b288c7c758eef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[96], dtype='int64'),
            ]


    class TestPrimitiveOp_1b877f488bd9bf193577eff551379ea6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e40b041dd714cf501c1b82e1156c262
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[48], dtype='int64'),
            ]


    class TestPrimitiveOp_2917cc5d772c91c481b11f43a32cb690(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb138e8ea9ebc6a55103396e5d894c2f
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], dtype='int64').reshape([24]),
            ]


    class TestPrimitiveOp_dc96071e7b13785a6386a685b462fafa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2478cc76adbd9d449b1a74b60c90c34
        def get_inputs(self):
            return [
                paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc96071e7b13785a6386a685b462fafa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2478cc76adbd9d449b1a74b60c90c34
        def get_inputs(self):
            return [
                paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_e6addcaf1c69ad70f6de7c32d245dcfe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8732, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b888ea9f4ed31319e118c7b130d38b6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6addcaf1c69ad70f6de7c32d245dcfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_8a956b726dab2721a784977507e53ff3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[256], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5214521906e396b767bfdc743ae6b865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a956b726dab2721a784977507e53ff3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_7bda16080ed280e6bb7816f1c8303d45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_60122ffd4eb2fc5ee94a65f12f7cf2e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_60122ffd4eb2fc5ee94a65f12f7cf2e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_1cdeeac3a309028d1851251b1bfc4d5f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7db5d633ec1c1d123725fc20de074a79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1cdeeac3a309028d1851251b1bfc4d5f
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4765012264251709, 0.2170112580060959, 0.4702689051628113, 0.4369574785232544, 0.244343563914299, 0.49530693888664246, 0.3121092915534973, 0.0963035523891449, 0.38352274894714355, 0.009155333042144775, 0.4616371691226959, 0.020641636103391647, 0.15587159991264343, 0.02491425909101963, 0.443154513835907, 0.24451406300067902], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_6bbf6cb3dd68f8ebf1bf4ada2d532761(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b41322013e40feea1fdc5a0eb2854c4
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_b47e083136bab1519432e12930ae9f22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b41322013e40feea1fdc5a0eb2854c4
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_d0a5d04052470b0d8d6a1caca8c64254(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(7581, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_802fe6cf4c2dbdee0f05c2b952759981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c9681b531783052c0e4d1094737f42e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_dece7d522d618b4505b77810c7a853a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4725, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_6bc86392b71c06f7b549688314280281(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3, 28, 28], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2152fde688f8afa43b51dcf59f84bbd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bc86392b71c06f7b549688314280281
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_0153e4557c71d082b53718bee695fa03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(577, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_473543387718b086f60d63fc9119bdbb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7c016d8a536c2cb9b3d6ee6f95108dce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_473543387718b086f60d63fc9119bdbb
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_155fe12cc4fbe5b1cd736be80e45d7a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_473543387718b086f60d63fc9119bdbb
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_53733b8b0d8892746f63d8385396331c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_224848fedf4e1be1fa2b5e14969fa3d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53733b8b0d8892746f63d8385396331c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_664cd07668eda31409c6a65077a1e762(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 4], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a48032a7c3a83cf81173f5aafba914d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_664cd07668eda31409c6a65077a1e762
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 4], dtype='int32'),
            ]


    
    class PrimitiveOp_49437b8c41f2179d2e65da92ebe3ee0f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b9ec8c88af43532a04c3d51d44c7a219(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49437b8c41f2179d2e65da92ebe3ee0f
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_0bbb3041759d72b09e91398405051c5d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 68], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a790b07413543f9b2100a100bc43ca35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bbb3041759d72b09e91398405051c5d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 68], dtype='int32'),
            ]


    
    class PrimitiveOp_32c478ba47c61965d209513e2c852396(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1723, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9d939f4d368035345bc76cbb6b0c9ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32c478ba47c61965d209513e2c852396
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_60019a16dfb3fd21213e20376c6b585f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1723, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7f6217ccbe4ebbd6a611cc5895d3b9ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60019a16dfb3fd21213e20376c6b585f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1723, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_81a4f7429755bb5416628ba6b8cdc5fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_473543387718b086f60d63fc9119bdbb
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0186bf9c3fe25f025061454de6bc085b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8400, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5214521906e396b767bfdc743ae6b865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a956b726dab2721a784977507e53ff3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_57a077c75e76b2ca8a7d7aab0cc719f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fafe3974c19dea491ba88de1fd22485
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_1e166390a59daa6df6f05aec6a7c6c93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d919a13f509de0d72f32a49750393c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_f5fb66a94dfd8a9e445bc1b28f3820b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b41322013e40feea1fdc5a0eb2854c4
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_6607080a9771f9242ca2d296e5f8335e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5004ac6c4b720e53d3370381ff2a8e64
        def get_inputs(self):
            return [
                paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6607080a9771f9242ca2d296e5f8335e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5004ac6c4b720e53d3370381ff2a8e64
        def get_inputs(self):
            return [
                paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5214521906e396b767bfdc743ae6b865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a956b726dab2721a784977507e53ff3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0ad0eafa5c3e79d8e997530d47a029f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(3549, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_47021b48726e126990ca53e682adabc9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 128, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2678c3fd09277fdd8fd2e53fca571a90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47021b48726e126990ca53e682adabc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7a85644279ccadb129ed9b88b6d97c0f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2cc658277e8698e149ff4319da6ffc1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a85644279ccadb129ed9b88b6d97c0f
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_1c2570e6d40ed29f9e5505f33143a253(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a2b98cbd2624974612a5a5731e0c2aa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c2570e6d40ed29f9e5505f33143a253
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_0fbdac91811e24202abf586e60ce19b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109, 4], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_af7a557db3fad5aab824a0171bdb6d11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fbdac91811e24202abf586e60ce19b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 4], dtype='int32'),
            ]


    
    class PrimitiveOp_8e8eed0fe84e969f3183916fcde8e936(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa1040663935e896afe4cd58249bf25c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8e8eed0fe84e969f3183916fcde8e936
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_4c4f93697c1863661353b7f81ff84333(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109, 68], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7d67a7af06145f4022f91a2ec9af41e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c4f93697c1863661353b7f81ff84333
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 68], dtype='int32'),
            ]


    
    class PrimitiveOp_df80db41a6f7049ae866a6e20675e8d8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5498, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_312b46a72e8850d7455c6d9a6206d809(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df80db41a6f7049ae866a6e20675e8d8
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dd13890ac39fe020cc3dca454f02bc32(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5498, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5c214763c347b420ab48b80172dfceb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd13890ac39fe020cc3dca454f02bc32
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5498, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_e1491a8982c4d888546e86043ef1cfce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_da22f3d9587fb4dde0fa0593c759b3fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1491a8982c4d888546e86043ef1cfce
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0b0952405ed8dcd78485899c1108f3b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_95190bf26f94c4a94c3be9580666ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b0952405ed8dcd78485899c1108f3b6
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bcf3aa8f2195df151329a6e16ce74754(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f675092cea2b25bd1f22e00a7a5aa1fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_964f9e01a90bc3e437cdb7b321fcc0c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(98, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_22fe7e9a82cc0ad3ed4ab7ca6e6fdc93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(99, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5214521906e396b767bfdc743ae6b865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a956b726dab2721a784977507e53ff3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    
    class PrimitiveOp_f1b69172e4584e1e25efe3e72c542237(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bee87742174786e5ce49ede2f9ecb341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1b69172e4584e1e25efe3e72c542237
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1a948e78f582b70a2b626d3c8dc11e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d9859c8b7984fe1f96fcf63c2094932
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_e1a948e78f582b70a2b626d3c8dc11e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d9859c8b7984fe1f96fcf63c2094932
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2156318ac36bb8e1ee334d298d3f2bff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(192, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_aa5295fe54ba22c79a4225da6cb61732(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_787adb78755723d7e5e9b29600300267(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa5295fe54ba22c79a4225da6cb61732
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cc09d1a4d7cea30b414a6b47329b4d78(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_078aa0195a1ef9d0489aebac3fb2e54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc09d1a4d7cea30b414a6b47329b4d78
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 192, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c9681b531783052c0e4d1094737f42e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_34bf0e5f3540cee2a538d457858d2f73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5214521906e396b767bfdc743ae6b865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a956b726dab2721a784977507e53ff3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_802fe6cf4c2dbdee0f05c2b952759981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c9681b531783052c0e4d1094737f42e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2678c3fd09277fdd8fd2e53fca571a90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47021b48726e126990ca53e682adabc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cc658277e8698e149ff4319da6ffc1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a85644279ccadb129ed9b88b6d97c0f
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c9681b531783052c0e4d1094737f42e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bcf3aa8f2195df151329a6e16ce74754(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_2906ce342347f06ebfbaef722ca7d6f4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_15cf04104e178a2986606d5187f83b90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2906ce342347f06ebfbaef722ca7d6f4
        def get_inputs(self):
            return [
                paddle.to_tensor([False, True, False, False, False, False], dtype='bool').reshape([6]),
            ]


    class TestPrimitiveOp_90a58c6d3c1d36d0a54a5a2a6e9bddd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2906ce342347f06ebfbaef722ca7d6f4
        def get_inputs(self):
            return [
                paddle.to_tensor([False, False, False, False, False, False], dtype='bool').reshape([6]),
            ]


    class TestPrimitiveOp_224848fedf4e1be1fa2b5e14969fa3d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53733b8b0d8892746f63d8385396331c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a48032a7c3a83cf81173f5aafba914d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_664cd07668eda31409c6a65077a1e762
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_b9ec8c88af43532a04c3d51d44c7a219(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49437b8c41f2179d2e65da92ebe3ee0f
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_026692e2dda9314a4414648a21cb07a6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 76], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a67eec9a7a267ba31a01f89e41b62ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_026692e2dda9314a4414648a21cb07a6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 76], dtype='int32'),
            ]


    
    class PrimitiveOp_448820bd0a44277c9b7491d01663c3cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1759, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd3735d4c9dee93e8bf921c627dd24b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_448820bd0a44277c9b7491d01663c3cd
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_515b2df4904b186d4d01c6fb7139e6e2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1759, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d63394b87ce068c416095ddf9c835563(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_515b2df4904b186d4d01c6fb7139e6e2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1759, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_6127b3c709676e1cb280f2c47ec37048(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2298f7e0a5691c59eff19a95ba076140(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6127b3c709676e1cb280f2c47ec37048
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4e7d2c7590a5d7be67a323618844565d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f2cd7f46c4e8e41084bd57b42f82ebc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e7d2c7590a5d7be67a323618844565d
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_1fd2c3d335b465ed28492c52cd2458d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 256, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7957468de5e890534175177e5458dd15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fd2c3d335b465ed28492c52cd2458d6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2cd7f46c4e8e41084bd57b42f82ebc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e7d2c7590a5d7be67a323618844565d
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_802fe6cf4c2dbdee0f05c2b952759981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0b0edc8276c10fc9d7bb0e027ab42527(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(28, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_de07304dbd124756772af50e24fac360(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(50, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f0e016916765bf7280f0b71372e47a19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_473543387718b086f60d63fc9119bdbb
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_182239e1ffe3ed9bf865d9aa1f21632e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4116, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_f783a15015ad25d87a4e92dfca33b627(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_709477bfaaac63c769df627bac3fb936(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f783a15015ad25d87a4e92dfca33b627
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cdfb5bf4f4a935e9ea0c02421e955400(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_179e55f6aa399ca539a75810dfeca2bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdfb5bf4f4a935e9ea0c02421e955400
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d31778eed4a7f24a47433b1207464fdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d839ae06b2506fad2dd389aee3e06599
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[80], dtype='int64'),
            ]


    class TestPrimitiveOp_68af7152486f41e6e844964237fedec5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84fd698027ded646cca2abe3206bc5df
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
            ]


    class TestPrimitiveOp_3bdfa9ffadc1c239c606848250aea563(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edef3039412121f63e84a43ce2c79182
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_6c14fd6497d9f17b9f0f2c6c3dd3d42f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bed2851ee43df48dcf93054fbbcf5e10
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c14fd6497d9f17b9f0f2c6c3dd3d42f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bed2851ee43df48dcf93054fbbcf5e10
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e75230232bf81471cad7146039bfab86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dd417b86875586a21858d0a4d4d6810
        def get_inputs(self):
            return [
                paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_9a82f129a8dc67e6cd1c6767dee0911a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a7986f9cb31e0df46a55ea01a7f50594(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a82f129a8dc67e6cd1c6767dee0911a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3306543529033661, 0.3834773600101471, 0.49611005187034607, 0.11812765151262283, 0.478644460439682, 0.32757896184921265, 0.3514584004878998, 0.3204523026943207, 0.48325031995773315, 0.25242266058921814, 0.4594441056251526, 0.10603760182857513, 0.09884601831436157, 0.039876293390989304, 0.12443646788597107, 0.20525671541690826, 0.031096970662474632, 0.08158884942531586, 0.2653093934059143, 0.3785844147205353, 0.4728078544139862, 0.47888636589050293, 0.37857428193092346, 0.030704200267791748], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_b09373898c6a4fac290ff3a9061d357d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb138e8ea9ebc6a55103396e5d894c2f
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
            ]


    class TestPrimitiveOp_34b860ae99925fcfc0756a8f169b4f92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb138e8ea9ebc6a55103396e5d894c2f
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
            ]


    class TestPrimitiveOp_3d54f5547e96de046763500ad795bcaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f4d0e63d97c505db8fff9e1194f2eb5
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6939955949783325], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1e166390a59daa6df6f05aec6a7c6c93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d919a13f509de0d72f32a49750393c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1e166390a59daa6df6f05aec6a7c6c93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d919a13f509de0d72f32a49750393c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_81e10633f9089dd38f84624695712084(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f34095942dc59cfe8a61aaf9cc0a6a1f
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_57a077c75e76b2ca8a7d7aab0cc719f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fafe3974c19dea491ba88de1fd22485
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_57a077c75e76b2ca8a7d7aab0cc719f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fafe3974c19dea491ba88de1fd22485
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_0fcdf590ec7a0638637cd0b5cb4f258c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba3b6992b7cf4500fd957ef9235401df
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb94b28ee4f98c79d4c881970a09b471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_78615fc6d63cf353c3760780a2a9baa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90140664290a2ebea0c9ccc3b00815bb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_78615fc6d63cf353c3760780a2a9baa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90140664290a2ebea0c9ccc3b00815bb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_b3a97447a766a3722ee7f4eb212d26e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8272d93ef23a7780bc3b71c27884c666
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9e89e028c6d753fa4b1735f1872f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c92e0ec02de5625fd55b34449d684532(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6069, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_b9ade000b56dfc9ad7764bac91135f44(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a45ca4e88068116e1c04c9f5ded921d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9ade000b56dfc9ad7764bac91135f44
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_c2582c8800f677fd439950c74c7aa317(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024, 4], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c3cec07ddfb85065b3c9e3d8a01a2e0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2582c8800f677fd439950c74c7aa317
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 4], dtype='int32'),
            ]


    
    class PrimitiveOp_42805152f4b0d0c5d6951a5998baa0c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_54d1394dae3a40f8c20ec10eea3e9f39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42805152f4b0d0c5d6951a5998baa0c7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_160a6e0bba7a428e31e95767cdbb1ed1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024, 68], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a1f717b884053cb039ea6f31ecc0be75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_160a6e0bba7a428e31e95767cdbb1ed1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 68], dtype='int32'),
            ]


    
    class PrimitiveOp_ecf8c4973ade45b58c65a9a419ca5c28(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1538, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_62513c01844fa9b5eff20308a19554cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecf8c4973ade45b58c65a9a419ca5c28
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9e610bd3801f0ebc6bc83354969ab4ba(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1538, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_698632b5b5ac462927d0fe11ac054aaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e610bd3801f0ebc6bc83354969ab4ba
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1538, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_224848fedf4e1be1fa2b5e14969fa3d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53733b8b0d8892746f63d8385396331c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_224848fedf4e1be1fa2b5e14969fa3d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53733b8b0d8892746f63d8385396331c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_1b86f4a2d7050aea7283c94296a1a729(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e31a11c132b74c1a48796b6d7c747bb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b86f4a2d7050aea7283c94296a1a729
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    class TestPrimitiveOp_3543022539aa86acc1acc8c5cc6a59a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dd417b86875586a21858d0a4d4d6810
        def get_inputs(self):
            return [
                paddle.to_tensor([8, 2], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_4b2f7fa85504601604bd28862b71284c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cbb0adbdb1b321c5623e0f764cb67afe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b2f7fa85504601604bd28862b71284c
        def get_inputs(self):
            return [
                paddle.to_tensor([0.004606406204402447, 0.23123939335346222, 0.4518365263938904, 0.139329195022583], dtype='float32').reshape([4]),
            ]


    
    class PrimitiveOp_1d5ebaf6ae78dcf7937064cde6b9685f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c0307cd3a9e9bfe5d7c34cdff18f404(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d5ebaf6ae78dcf7937064cde6b9685f
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_a1bdc2304e986d99a425dace3ddb2191(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d5ebaf6ae78dcf7937064cde6b9685f
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_904e9b9355d7e13194363e52c0f2f695(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(52, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b5489bc08cb69df0cd70fb6389691632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(202, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5214521906e396b767bfdc743ae6b865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a956b726dab2721a784977507e53ff3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_5214521906e396b767bfdc743ae6b865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a956b726dab2721a784977507e53ff3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_bb8d3995b3b5a551c656ee8b23ee0c42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1025, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5214521906e396b767bfdc743ae6b865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a956b726dab2721a784977507e53ff3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_08a93a65b2fdf48cb93dcd473854c5a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6472103106d2b18ba79b111b44349ece
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype='int64').reshape([14]),
            ]


    class TestPrimitiveOp_2e739c1531583ca947b98d424070081f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecc23effe974611db28e250e4c3e5354
        def get_inputs(self):
            return [
                paddle.uniform([14, 14, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3577090808847984eb3185a3b1a2c675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c527f1a745b53c2f00135a1a3e45cd64
        def get_inputs(self):
            return [
                paddle.uniform([14, 14, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c15075ac2b0f4dc3047a15b0345dfe00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b6bb2ed6f5ff2db0d3b174c6413b7cb
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], dtype='int64').reshape([28]),
            ]


    class TestPrimitiveOp_ff7406548ba643fb37280e55696548a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86a5a8a79c6c5b9e03d5b869e5e8a103
        def get_inputs(self):
            return [
                paddle.uniform([28, 28, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_afb7f5442fbfbc5be5bce5f1d0cc1943(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a2214949df8e1cf8049f839e7a30e80b
        def get_inputs(self):
            return [
                paddle.uniform([28, 28, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d798a4a46ab85834fa927ba694fd9d98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de7c140b7de9b0938673704688fbbb95
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[56], dtype='int64'),
            ]


    class TestPrimitiveOp_c827c6eb665d82c004f8be91bc26ed6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce1a67b3b1ff3815092f9cb98d9b8434
        def get_inputs(self):
            return [
                paddle.uniform([56, 56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96ffc4db00ce359c0afed1c8469bbbcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13e797d8acd75d28933cfd7c6ce383fc
        def get_inputs(self):
            return [
                paddle.uniform([56, 56, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_60122ffd4eb2fc5ee94a65f12f7cf2e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_60122ffd4eb2fc5ee94a65f12f7cf2e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_de5243416defd095a8f4901acb914079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(104, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_de5243416defd095a8f4901acb914079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(104, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_ccb783b521266f2f8cf41d840bb66cd1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d1a2a0ae7ee98fdc09a0af9b99825da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccb783b521266f2f8cf41d840bb66cd1
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_3d1a2a0ae7ee98fdc09a0af9b99825da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccb783b521266f2f8cf41d840bb66cd1
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_02d2a95467c41ba8add997efa455bcd9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3f481a637255430e708e2b914e4ddfd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02d2a95467c41ba8add997efa455bcd9
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10cdb2ae7d8bba2fdab34a87674d6a63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e84ab84058793cee2719f0ee499707f4
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5be6741a1a32bd81e70e6152de3c2ebf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e84ab84058793cee2719f0ee499707f4
        def get_inputs(self):
            return [
                paddle.to_tensor(7, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5214521906e396b767bfdc743ae6b865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a956b726dab2721a784977507e53ff3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5214521906e396b767bfdc743ae6b865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a956b726dab2721a784977507e53ff3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8f098e1c95c87764c32c49b9c17ee070(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba4ce69d03b2589504a7449fa145012
        def get_inputs(self):
            return [
                paddle.to_tensor([300.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8d90bf6e0d133c3ffc11ca0b0d295d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_473543387718b086f60d63fc9119bdbb
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f0e016916765bf7280f0b71372e47a19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_473543387718b086f60d63fc9119bdbb
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3d1a2a0ae7ee98fdc09a0af9b99825da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccb783b521266f2f8cf41d840bb66cd1
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_b5709967a606a9829dba44e9c73fffdb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116, 4], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3c3e7962bc05219b22ce712e1f26f306(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5709967a606a9829dba44e9c73fffdb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 4], dtype='int32'),
            ]


    
    class PrimitiveOp_7f0d91856376122ee22f4fee9faec2ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_67d01b959536b65fde2addaea4304442(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f0d91856376122ee22f4fee9faec2ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_5b6edf8984c796df01e63fb9f1a3fed2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116, 68], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_50c67633b7cb012d27d61c82888d872f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b6edf8984c796df01e63fb9f1a3fed2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 68], dtype='int32'),
            ]


    
    class PrimitiveOp_c0ff25ed39e6b3cdc7e30fac3c9da9c0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2135, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4835fb0b9aa61982bd9423286b7e496b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0ff25ed39e6b3cdc7e30fac3c9da9c0
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3ce7be503e77fae58a8ef004cf644942(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2135, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_18c2065110e0d1293dc650307f46b8c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ce7be503e77fae58a8ef004cf644942
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2135, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5214521906e396b767bfdc743ae6b865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a956b726dab2721a784977507e53ff3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_1f617b341f517a443d6c09a453fc7f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(14, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8d063c09f60fcc9258d34e3e9b863cd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(25, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_7ea3328d9a326a41a2976895811f64e0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8078450105e7dfd1872b5fd58aeb08e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ea3328d9a326a41a2976895811f64e0
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_3c14ee42ac93c0c79fd81035c048745e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261, 4], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_316a7b3276d233f47765cb1db32b02ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3c14ee42ac93c0c79fd81035c048745e
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 4], dtype='int32'),
            ]


    
    class PrimitiveOp_0593aacb58bba1bb2fce727d8df3cd9d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7a55526c226d951ec1de5d8df7256c60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0593aacb58bba1bb2fce727d8df3cd9d
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_0490f563a8e5834d568424194e85f4af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261, 68], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c02e53ce39c88e1ef9227cc3afae37f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0490f563a8e5834d568424194e85f4af
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 68], dtype='int32'),
            ]


    
    class PrimitiveOp_07bf12b43c4cd10342d6909e76f48208(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4590, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_58f409ce7c2bd7aee1fffe35b69205b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07bf12b43c4cd10342d6909e76f48208
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_afb61c5a69760a31bee5007fcc1c1c43(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4590, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f7f7d9ddeef63c1dfa67378417295d52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afb61c5a69760a31bee5007fcc1c1c43
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4590, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_72144a064a8226e0012776d232e7bd49(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 28, 28], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a7e5011631b2dbc0ba937cb5baabea3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72144a064a8226e0012776d232e7bd49
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[6, 28, 28], dtype='int32'),
            ]


    
    class PrimitiveOp_8c9ee497320cde433b3a96941bc7248e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2434, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0ae83209e2ad3dd633a73bc944be3942(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c9ee497320cde433b3a96941bc7248e
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d15a7cee783a544c3f85961383bec90a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7af1227d0f5a1e0abf18bee01d5e23e
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_f53add18a25889f5b64d19c5fadc5c87(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100, 4], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e65269cebc828622215d59b8aa35586d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f53add18a25889f5b64d19c5fadc5c87
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 4], dtype='int32'),
            ]


    
    class PrimitiveOp_f139ce3be13dc28547eed47fb37de7b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b1e37af7244103bc61b5c822eb352ca5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f139ce3be13dc28547eed47fb37de7b5
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_e06f62068a004e7b0911232194973094(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100, 68], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e951a266d9ee15f076bff77074309b19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e06f62068a004e7b0911232194973094
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 68], dtype='int32'),
            ]


    
    class PrimitiveOp_1430e251bf837bff2e46d8b5e48349d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1042, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a8db749a1271a8739434a502e9cf8cbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1430e251bf837bff2e46d8b5e48349d5
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f4cec4f8fcb634ee242b1dd47ebfc9d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1042, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_625b84f962ec4b81501a7fb783779617(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cec4f8fcb634ee242b1dd47ebfc9d5
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1042, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5214521906e396b767bfdc743ae6b865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a956b726dab2721a784977507e53ff3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_159853c5be6e7ff71d2a2861bf97f1ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(9261, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bcf3aa8f2195df151329a6e16ce74754(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0ad58846f902324f75475abf194f1553(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e7d9a4859786add76a55da68a87b5f53
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[68], dtype='int64'),
            ]


    class TestPrimitiveOp_3d0fb47adbe2e30f6e74c529643e5519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ade11628c930b53312f35cea2b97b307
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[34], dtype='int64'),
            ]


    class TestPrimitiveOp_6ce018a44e36dfda64be1cd920c8c296(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6773edf339f76fbc72008ad854b1bd67
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype='int64').reshape([17]),
            ]


    class TestPrimitiveOp_5e43b2fad1f08f34b6e0ff928674cd68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1dbebddafbd7d04dc39bd6f57b1cf78
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e43b2fad1f08f34b6e0ff928674cd68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1dbebddafbd7d04dc39bd6f57b1cf78
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_da22f3d9587fb4dde0fa0593c759b3fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1491a8982c4d888546e86043ef1cfce
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95190bf26f94c4a94c3be9580666ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b0952405ed8dcd78485899c1108f3b6
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_da22f3d9587fb4dde0fa0593c759b3fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1491a8982c4d888546e86043ef1cfce
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95190bf26f94c4a94c3be9580666ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b0952405ed8dcd78485899c1108f3b6
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_da22f3d9587fb4dde0fa0593c759b3fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1491a8982c4d888546e86043ef1cfce
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95190bf26f94c4a94c3be9580666ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b0952405ed8dcd78485899c1108f3b6
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_eb531f5775a9f575efca9cb6a04c768b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(2048, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_b9f3c1253e24697534fd18a51e2200d8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e054c80b38108bc6334b40782cf04414(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9f3c1253e24697534fd18a51e2200d8
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_155ae952b7c83c06b0e6608c7d912b9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 1, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_96500e47a35c3d89079326abd670909b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_155ae952b7c83c06b0e6608c7d912b9e
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2048, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_78615fc6d63cf353c3760780a2a9baa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90140664290a2ebea0c9ccc3b00815bb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_78615fc6d63cf353c3760780a2a9baa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90140664290a2ebea0c9ccc3b00815bb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_b3a97447a766a3722ee7f4eb212d26e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8272d93ef23a7780bc3b71c27884c666
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9e89e028c6d753fa4b1735f1872f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_57a077c75e76b2ca8a7d7aab0cc719f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fafe3974c19dea491ba88de1fd22485
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_57a077c75e76b2ca8a7d7aab0cc719f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fafe3974c19dea491ba88de1fd22485
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_0fcdf590ec7a0638637cd0b5cb4f258c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba3b6992b7cf4500fd957ef9235401df
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb94b28ee4f98c79d4c881970a09b471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1e166390a59daa6df6f05aec6a7c6c93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d919a13f509de0d72f32a49750393c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1e166390a59daa6df6f05aec6a7c6c93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d919a13f509de0d72f32a49750393c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_81e10633f9089dd38f84624695712084(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f34095942dc59cfe8a61aaf9cc0a6a1f
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f5fb66a94dfd8a9e445bc1b28f3820b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b41322013e40feea1fdc5a0eb2854c4
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f5fb66a94dfd8a9e445bc1b28f3820b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b41322013e40feea1fdc5a0eb2854c4
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
            ]


    
    class PrimitiveOp_f3998fead08cdeb31750db7b603b6860(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 16, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ecff19e8ba7b812387e063e34bfb853(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f3998fead08cdeb31750db7b603b6860
        def get_inputs(self):
            return [
                paddle.uniform([16, 16, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_7409b83bb1b2250804e2e3234f44d640(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f81d369f8267bf89f510a30c2342fb5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7409b83bb1b2250804e2e3234f44d640
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([8]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f81d369f8267bf89f510a30c2342fb5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7409b83bb1b2250804e2e3234f44d640
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([8]),
            ]


    
    class PrimitiveOp_37aa820851d7bfb351e7831f0243a0fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 8, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_99e7aceeab80cff13d177a9e959dd1fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37aa820851d7bfb351e7831f0243a0fe
        def get_inputs(self):
            return [
                paddle.uniform([8, 8, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e244c47b82deaf9f39c116ef0e74df1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(2100, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_bce5f35d8abc0f8f2e58058328db1548(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b683c002822df46f5a71f5865f17db58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bce5f35d8abc0f8f2e58058328db1548
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95190bf26f94c4a94c3be9580666ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b0952405ed8dcd78485899c1108f3b6
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b683c002822df46f5a71f5865f17db58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bce5f35d8abc0f8f2e58058328db1548
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95190bf26f94c4a94c3be9580666ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b0952405ed8dcd78485899c1108f3b6
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b683c002822df46f5a71f5865f17db58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bce5f35d8abc0f8f2e58058328db1548
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95190bf26f94c4a94c3be9580666ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b0952405ed8dcd78485899c1108f3b6
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_eb531f5775a9f575efca9cb6a04c768b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(2048, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_ecec1b53a99a0d35295ed5fd21753aa9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd80bbdd001b92dbc35f6527e7792495(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecec1b53a99a0d35295ed5fd21753aa9
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96500e47a35c3d89079326abd670909b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_155ae952b7c83c06b0e6608c7d912b9e
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2048, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8d3995b3b5a551c656ee8b23ee0c42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1025, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_209f5f1c57d91ed00d504af92bfbf4b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b2f07c5d37dfd4bd5a02662e1744b4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_209f5f1c57d91ed00d504af92bfbf4b5
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_b1b233d9ab04ff9cec56ceadb9f6ebd1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725, 4], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_63364423da8fd7a60a0f1477a7e14747(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1b233d9ab04ff9cec56ceadb9f6ebd1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 4], dtype='int32'),
            ]


    
    class PrimitiveOp_f0573a5a3c2f2168b4467b7c236edb21(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_159e150f81ed6dd1468bac35e9b327e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0573a5a3c2f2168b4467b7c236edb21
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_ca4227577154780fc41d310a246da5f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725, 68], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_03db0d54eada1cb842f0a03541b173b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca4227577154780fc41d310a246da5f0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 68], dtype='int32'),
            ]


    
    class PrimitiveOp_c5f3fdfdf63b85703def2c861295b490(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2339, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c1c919ee802d555d7ca41be85141ba10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5f3fdfdf63b85703def2c861295b490
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ce62601fe1d93b4a8ecb7b52892c1651(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2339, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_402a99985cf4a53c1ff042a8945c2c6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce62601fe1d93b4a8ecb7b52892c1651
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2339, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_802fe6cf4c2dbdee0f05c2b952759981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_c585c64f4b92edd3ce6f8bdba1b31a7b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_10415e99ffc96049e0a57a5fddf25f6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c585c64f4b92edd3ce6f8bdba1b31a7b
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_5222079ab6abf0c8fefaae4137f3b6a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069, 4], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c9478fcd8a83610cf192d3ace8f9e140(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5222079ab6abf0c8fefaae4137f3b6a7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 4], dtype='int32'),
            ]


    
    class PrimitiveOp_29a180bb60e781882f66e3c43c4ee8e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_17b925ebeeb77449045a9f4ff0671e04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29a180bb60e781882f66e3c43c4ee8e6
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_7f58c106a97212fcf573e462d0c6cf79(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069, 68], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_792e36e7239a55d3a92c88682f7830dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f58c106a97212fcf573e462d0c6cf79
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 68], dtype='int32'),
            ]


    
    class PrimitiveOp_ea0a64e0ece951976b91a7025558cd44(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3063, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_671310444a6c0d9a5b59dbd611caa100(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea0a64e0ece951976b91a7025558cd44
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a0a682d7523e117a94bf970e803f4f0b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3063, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3b1f7d2beb2254a712e88a1119860255(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0a682d7523e117a94bf970e803f4f0b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3063, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_2e2a03a73deef2ebb46ccb727d6916b7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_54ef4d401934f0bcef9c88d4eaac7937(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e2a03a73deef2ebb46ccb727d6916b7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_9e1125786813482ac0512203787456f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581, 4], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5218edd4e483772ca406ed940b1656fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e1125786813482ac0512203787456f5
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 4], dtype='int32'),
            ]


    
    class PrimitiveOp_eddaf49d4449917ef10eee9c7a6d7b05(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_acebf7e828d3ee9949c8832da65fc55c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eddaf49d4449917ef10eee9c7a6d7b05
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_1b666ed7591bf789816252ad25852f3d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581, 68], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8692b9575e5ab31303bc0ebb627614e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b666ed7591bf789816252ad25852f3d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 68], dtype='int32'),
            ]


    
    class PrimitiveOp_db5d3b57b95fc9d15cab4169c4097b50(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3822, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0bb2ac1026080029fa18d90cc1b7686c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db5d3b57b95fc9d15cab4169c4097b50
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f5ccc56dcb1338e879ea35b7653b1851(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3822, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_355cfa584a6fbb6e3d27db2dff2b0333(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5ccc56dcb1338e879ea35b7653b1851
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3822, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6a3f0452564d385d91569fde79419392(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba4ce69d03b2589504a7449fa145012
        def get_inputs(self):
            return [
                paddle.to_tensor([100.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_cca72d2a6439ce40b2a2d3b7b6c0a4d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11109, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f319b5712396bf56b42629c11e3981fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_473543387718b086f60d63fc9119bdbb
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_6c4ad354ca14bee15248f7f83c5df981(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 28, 28], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d589502a20c49cbdeda7159653644c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c4ad354ca14bee15248f7f83c5df981
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_bfa8aaf3ceda90e4b60cab96d0758ad9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_edbbe79e5dfef890a0a02ea2891d500b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([11], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e9bb4cc7be02a14ac6eea0421d4d2463(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([384], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bdf72173d9ed3edb7112612db5c3d998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([28], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f51299282ffee062040fa3d2b67aeb38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([77], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_fc1175cfddfcd5e5bfd8590d9f14a0be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc9232f8ccf5abe5be92c2749fdc3a6c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[152], dtype='int64'),
            ]


    class TestPrimitiveOp_6878171d55b6bba9cc9f848db81eac08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ed02a1c76d46a1e5630efd7b9502950
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[100], dtype='int64'),
            ]


    class TestPrimitiveOp_6f5db86631ae22d24a550eadb1e4f5b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27430bd5e8a381a6c05a561dae8575cf
        def get_inputs(self):
            return [
                paddle.uniform([100, 152, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ebd95ae730efae30290b72b71ead83a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461554130b035570edf5be277ff9cfd8
        def get_inputs(self):
            return [
                paddle.uniform([100, 152, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98ca641887034778e0d3e89b44a003c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4cd883e14789757030af4095b78018b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[76], dtype='int64'),
            ]


    class TestPrimitiveOp_765ea9296bf58e935e728be4a49c723a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a973a591c8ee767df009b149c504545d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[50], dtype='int64'),
            ]


    class TestPrimitiveOp_33433d7192c0af9df49c49da8351a45e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76c708e82f7438fe00d117e99cb675aa
        def get_inputs(self):
            return [
                paddle.uniform([50, 76, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6070d6412a2bee9e49cab59949d1642e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4795b870d1c1d9292ce5a5bd32164fff
        def get_inputs(self):
            return [
                paddle.uniform([50, 76, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1207e13bfdcda2dac72c83cc2d171fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37f23af4254b3e668dbcb1353c7a9ab2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[38], dtype='int64'),
            ]


    class TestPrimitiveOp_2ca51e31b1e75a5e1dc09603e8d95188(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b62c0a086d1a37da5b737fa9a21df82
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype='int64').reshape([25]),
            ]


    class TestPrimitiveOp_7671b3728a440a899bcac3b7f102a19a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb103be57129aa533357a57f599d0847
        def get_inputs(self):
            return [
                paddle.uniform([25, 38, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_989e0aa9a98729509c80c4589329c6d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81fb568efc0137a67b6dcf28fb6fd715
        def get_inputs(self):
            return [
                paddle.uniform([25, 38, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a1e1375f66f2801ae2d952bb3f17d96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e51fc23940a6b12d25b5badc3dd697fb
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype='int64').reshape([19]),
            ]


    class TestPrimitiveOp_67d24f4eb63084622ce12444c999a9f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8bf18062474d6c43ae826be76808706
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype='int64').reshape([13]),
            ]


    class TestPrimitiveOp_f85b71fe4b3c9e490193fe8c05887450(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05b29b433ab0937e8fe16428b44dba47
        def get_inputs(self):
            return [
                paddle.uniform([13, 19, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aee9d3190043e7fc7d92fbdcd7593de6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95022d4d291260aa4d534f4364e777c5
        def get_inputs(self):
            return [
                paddle.uniform([13, 19, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca3214e01eb4e38c3d02a439039c6364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eaa4743693b05cf99345cfaa026a990b
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64').reshape([10]),
            ]


    class TestPrimitiveOp_a8afb99ac3c5c6ec74a4daa1e03e5d2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_543120e7a9a61eff3ed407a860346bec
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6], dtype='int64').reshape([7]),
            ]


    class TestPrimitiveOp_186427a771edd8be4562186d8ceeb782(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62437164d4281759d954b859874051ee
        def get_inputs(self):
            return [
                paddle.uniform([7, 10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_175f7cfaba243fb79d6e39220e742e58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e35b7f1a5582b4528f9ed2c763190bb2
        def get_inputs(self):
            return [
                paddle.uniform([7, 10, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_e09bd494a0b79a264bef22671cfcec80(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ae0e644369664c27a5dc3b983ddb13a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e09bd494a0b79a264bef22671cfcec80
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_179e55f6aa399ca539a75810dfeca2bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdfb5bf4f4a935e9ea0c02421e955400
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_f10aaa84e04daf1b65b61af32e196036(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_94b0e1784d94b7f3592edebe94a202f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f10aaa84e04daf1b65b61af32e196036
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2cd7f46c4e8e41084bd57b42f82ebc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e7d2c7590a5d7be67a323618844565d
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c9681b531783052c0e4d1094737f42e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_d0bae62795ca5e11ee85dbc7ace9ecac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_25cac92f0351606b72fae319c4389978(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0bae62795ca5e11ee85dbc7ace9ecac
        def get_inputs(self):
            return [
                paddle.to_tensor([0.34141385555267334, 0.08070738613605499, 0.03860168904066086, 0.49608147144317627, 0.41707998514175415, 0.07405710965394974, 0.39049777388572693, 0.14312376081943512, 0.4575801193714142, 0.44072431325912476, 0.48256993293762207, 0.10408809781074524, 0.3957173824310303, 0.045056845992803574, 0.2210846096277237, 0.11286043375730515, 0.109720878303051, 0.17332060635089874, 0.47122129797935486, 0.48317795991897583], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_a0b2ef11b87fdf38aca06fbf8e9473b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edef3039412121f63e84a43ce2c79182
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_c2274c0a4084c27ed8731ad574803aa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edef3039412121f63e84a43ce2c79182
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1e166390a59daa6df6f05aec6a7c6c93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d919a13f509de0d72f32a49750393c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1e166390a59daa6df6f05aec6a7c6c93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d919a13f509de0d72f32a49750393c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_81e10633f9089dd38f84624695712084(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f34095942dc59cfe8a61aaf9cc0a6a1f
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_57a077c75e76b2ca8a7d7aab0cc719f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fafe3974c19dea491ba88de1fd22485
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_57a077c75e76b2ca8a7d7aab0cc719f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fafe3974c19dea491ba88de1fd22485
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_0fcdf590ec7a0638637cd0b5cb4f258c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba3b6992b7cf4500fd957ef9235401df
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb94b28ee4f98c79d4c881970a09b471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_78615fc6d63cf353c3760780a2a9baa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90140664290a2ebea0c9ccc3b00815bb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_78615fc6d63cf353c3760780a2a9baa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90140664290a2ebea0c9ccc3b00815bb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_b3a97447a766a3722ee7f4eb212d26e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8272d93ef23a7780bc3b71c27884c666
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9e89e028c6d753fa4b1735f1872f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8d90bf6e0d133c3ffc11ca0b0d295d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_473543387718b086f60d63fc9119bdbb
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3d1a2a0ae7ee98fdc09a0af9b99825da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccb783b521266f2f8cf41d840bb66cd1
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_3c3e7962bc05219b22ce712e1f26f306(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5709967a606a9829dba44e9c73fffdb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_67d01b959536b65fde2addaea4304442(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f0d91856376122ee22f4fee9faec2ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_50c67633b7cb012d27d61c82888d872f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b6edf8984c796df01e63fb9f1a3fed2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 68], dtype='int32'),
            ]


    
    class PrimitiveOp_286a2d96693d9335d6ce742719301566(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2057, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c2f77cb2a0c7244e7a2f9cc1ae162289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_286a2d96693d9335d6ce742719301566
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_63a5f0fc1bfa8583192928703b563a87(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2057, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8d524733c6fce741c3cd782575e14ea8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63a5f0fc1bfa8583192928703b563a87
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2057, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_1be48382e1f332147bcc4739f68f8141(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 97, 97], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d54030b7720ddb59ca94bed47da0ce5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1be48382e1f332147bcc4739f68f8141
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95190bf26f94c4a94c3be9580666ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b0952405ed8dcd78485899c1108f3b6
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0af5f14fbb5c825ceafed43ed9515b50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(3024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c016d8a536c2cb9b3d6ee6f95108dce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_473543387718b086f60d63fc9119bdbb
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_95bd444b57f07058e3fb7d2baece734b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c1b262b9184613bdc87720c8f1ff057
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[72], dtype='int64'),
            ]


    class TestPrimitiveOp_e1a948e78f582b70a2b626d3c8dc11e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d9859c8b7984fe1f96fcf63c2094932
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_b8420ccb25952117d49fdc7f4a3f2277(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5f14d69836727d2e3c01e8580db0dd6
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype='int64').reshape([18]),
            ]


    class TestPrimitiveOp_495f694623b01c30c75fb2c8f4f790cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6514caaea4dfe18793e38eeb8c30609d
        def get_inputs(self):
            return [
                paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_495f694623b01c30c75fb2c8f4f790cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6514caaea4dfe18793e38eeb8c30609d
        def get_inputs(self):
            return [
                paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5214521906e396b767bfdc743ae6b865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a956b726dab2721a784977507e53ff3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_ea0df210b00f923541bf6bcc99305f0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1174, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5ee52e144167a1efe8c05d86d3f96e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84a6c40e9e96914b49308ff1eec6b910
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e9c329c6bb6119fae04cdd828fe8bea6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_473543387718b086f60d63fc9119bdbb
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b683c002822df46f5a71f5865f17db58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bce5f35d8abc0f8f2e58058328db1548
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95190bf26f94c4a94c3be9580666ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b0952405ed8dcd78485899c1108f3b6
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8d90bf6e0d133c3ffc11ca0b0d295d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_473543387718b086f60d63fc9119bdbb
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_709477bfaaac63c769df627bac3fb936(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f783a15015ad25d87a4e92dfca33b627
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_179e55f6aa399ca539a75810dfeca2bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdfb5bf4f4a935e9ea0c02421e955400
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_fdefb4aad294bf75413975654319a45f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ac9d36413f185c99ac930c289aab4523(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdefb4aad294bf75413975654319a45f
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_b3b064d073ff82e989c65dc286c1f5e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400, 4], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ece720de14f148afa7fa862b71d47b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3b064d073ff82e989c65dc286c1f5e6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 4], dtype='int32'),
            ]


    
    class PrimitiveOp_de6daf50a7054edad93c7866e05855f7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a3a7a6007ca46a0b3773415e7010b1c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de6daf50a7054edad93c7866e05855f7
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_8859c843b5e567fdbf3950556752f28b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400, 68], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a6542c1a677103cab85209e567e23010(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8859c843b5e567fdbf3950556752f28b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 68], dtype='int32'),
            ]


    
    class PrimitiveOp_70accc10763024e6f87c009c43ee7e47(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4189, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2fa189abde45da3f43f622440d01563f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70accc10763024e6f87c009c43ee7e47
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_95b6e3b541358e2387aa22003ac70215(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4189, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e4eecb132ff97e1cc1280fffc8b779cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95b6e3b541358e2387aa22003ac70215
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4189, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_ea0df210b00f923541bf6bcc99305f0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1174, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5214521906e396b767bfdc743ae6b865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a956b726dab2721a784977507e53ff3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1e166390a59daa6df6f05aec6a7c6c93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d919a13f509de0d72f32a49750393c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1e166390a59daa6df6f05aec6a7c6c93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d919a13f509de0d72f32a49750393c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_81e10633f9089dd38f84624695712084(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f34095942dc59cfe8a61aaf9cc0a6a1f
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_57a077c75e76b2ca8a7d7aab0cc719f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fafe3974c19dea491ba88de1fd22485
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_57a077c75e76b2ca8a7d7aab0cc719f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fafe3974c19dea491ba88de1fd22485
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_0fcdf590ec7a0638637cd0b5cb4f258c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba3b6992b7cf4500fd957ef9235401df
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb94b28ee4f98c79d4c881970a09b471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_78615fc6d63cf353c3760780a2a9baa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90140664290a2ebea0c9ccc3b00815bb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_78615fc6d63cf353c3760780a2a9baa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90140664290a2ebea0c9ccc3b00815bb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_b3a97447a766a3722ee7f4eb212d26e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8272d93ef23a7780bc3b71c27884c666
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9e89e028c6d753fa4b1735f1872f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_fcf1f3382b59bc773303e88b5072340e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_489fd001f1dce121f475610cc66041e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcf1f3382b59bc773303e88b5072340e
        def get_inputs(self):
            return [
                paddle.to_tensor([300.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0ad0eafa5c3e79d8e997530d47a029f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(3549, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    
    class PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0caf4225bb7d43f2f45a1fa28f10b6b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_2be5d6458dd1cacb5194c5338c64b953(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb94b28ee4f98c79d4c881970a09b471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_c4b3c17c2c1522493c2c36d22ceb6535(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9e89e028c6d753fa4b1735f1872f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f89d72b0790c24d0db50e2580374a497(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32a19618a8f19392b62d9a801d8cac68
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_5941bc0de4eba289d0e5be4b4f2214b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcf1f3382b59bc773303e88b5072340e
        def get_inputs(self):
            return [
                paddle.to_tensor([100.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_67046a6efbf3a7a1bd2ad9436cd0f103(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_67046a6efbf3a7a1bd2ad9436cd0f103(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_6e47b6d269e54aa74ea62513a14d6489(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89ef0551feeece2d18d122d3342cab46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e47b6d269e54aa74ea62513a14d6489
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            ]


    class TestPrimitiveOp_7bd314659b640e302547c0b9e9d89b7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d69ec1bcb65db19225d44841159e54d2
        def get_inputs(self):
            return [
                paddle.to_tensor([128], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9b42d248e61a96189607e41e1be5504e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d69ec1bcb65db19225d44841159e54d2
        def get_inputs(self):
            return [
                paddle.to_tensor([16], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e8df93d5e317d77438ed6c60e8173dbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d69ec1bcb65db19225d44841159e54d2
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_849deb9444cd457a9d7e9ce892a47120(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[96], dtype='int64'),
            ]


    class TestPrimitiveOp_d7c4ef02ee8133f61f4bbe09e278f6f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[48], dtype='int64'),
            ]


    class TestPrimitiveOp_322a78a92344794bad310970132bc604(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], dtype='int64').reshape([24]),
            ]


    
    class PrimitiveOp_77938fc03a986310c6c2dddb16ea348e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e64d41108ce05b97dd6a1cd18c5516a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77938fc03a986310c6c2dddb16ea348e
        def get_inputs(self):
            return [
                paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e64d41108ce05b97dd6a1cd18c5516a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77938fc03a986310c6c2dddb16ea348e
        def get_inputs(self):
            return [
                paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0f763c3b59876e2220b915dea7b16ad9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ac4ca313e7fbd3c10e159cd7bd7c0be
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_b31995b8aedb34b8ce56a4ddfbbba328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d69ec1bcb65db19225d44841159e54d2
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_60122ffd4eb2fc5ee94a65f12f7cf2e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_60122ffd4eb2fc5ee94a65f12f7cf2e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f73b1c9686b81e23bad03d63d4117473(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895e28dbe52d7a21dec773cd0e0db85a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4765012264251709, 0.2170112580060959, 0.4702689051628113, 0.4369574785232544, 0.244343563914299, 0.49530693888664246, 0.3121092915534973, 0.0963035523891449, 0.38352274894714355, 0.009155333042144775, 0.4616371691226959, 0.020641636103391647, 0.15587159991264343, 0.02491425909101963, 0.443154513835907, 0.24451406300067902], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_55f22f83b00b1b428aed96fa4c79ec74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_bb5d9aa93516f85a282dce087895dccf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_d0a5d04052470b0d8d6a1caca8c64254(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(7581, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_802fe6cf4c2dbdee0f05c2b952759981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c9681b531783052c0e4d1094737f42e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_dece7d522d618b4505b77810c7a853a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4725, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d70e46bb12ce27d57cd853881bd577cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32a19618a8f19392b62d9a801d8cac68
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_0153e4557c71d082b53718bee695fa03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(577, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ef1ccc76a8035835ee9d8a5de828c7d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_87c9b2cdd6d12ef417ebde9610eb701e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_32904982dc10676823c0da88bc0a8ed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5b801c6443b883acfc88853e629b6b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 4], dtype='int32'),
            ]


    
    class PrimitiveOp_9c6ab8826c46f53673d2600abe55eefe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_16a6891af27d79c6a00c1078c9671f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c6ab8826c46f53673d2600abe55eefe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_03d9f58c3e15b860366cb3762ca3c1b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 68], dtype='int32'),
            ]


    
    class PrimitiveOp_a4d325e19128a63c2f05ee741ce186c0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_56026609bc5cb98dee2d38e0ae57bffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4d325e19128a63c2f05ee741ce186c0
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dcdb8b145ad4f0d69dd06e27cbec8065(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_abc263abfed10c52a7f2ac7d0664c4b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcdb8b145ad4f0d69dd06e27cbec8065
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1723, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_6fe7aa1a1a3cb0946799831807fa632e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0186bf9c3fe25f025061454de6bc085b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8400, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_617dab664ac3ba7b80a678548a4279e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_510b97c7ee7fd5a97569defa78e86e37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77938fc03a986310c6c2dddb16ea348e
        def get_inputs(self):
            return [
                paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_510b97c7ee7fd5a97569defa78e86e37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77938fc03a986310c6c2dddb16ea348e
        def get_inputs(self):
            return [
                paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0ad0eafa5c3e79d8e997530d47a029f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(3549, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_012a6fae748b97b18b09071037379250(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_77c05b27af938dada5589a07edf0f3e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d146990ffb593c3ee23bc0a92d4e7488(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_397d2618b41e417fc6014a078fc53395(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_9cf345acb6d072cef1ae301446def235(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c6ab8826c46f53673d2600abe55eefe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_80730229b8e5257aa49e5fa954d7b04b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_3717abd73d6fcc4c5e707f39237b7b04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4d325e19128a63c2f05ee741ce186c0
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e760d90bcaa61f72696028ef918c2223(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcdb8b145ad4f0d69dd06e27cbec8065
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5498, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1f29ca91e7986af780fd3cc740eda9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2257ef4974735a200c547013f17cd828(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bcf3aa8f2195df151329a6e16ce74754(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f675092cea2b25bd1f22e00a7a5aa1fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_964f9e01a90bc3e437cdb7b321fcc0c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(98, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_22fe7e9a82cc0ad3ed4ab7ca6e6fdc93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(99, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_dba0fe45cf9a305cc7dbcc5b36c5d148(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895e28dbe52d7a21dec773cd0e0db85a
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3fa446a52be134607b410e6bfa9b7946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_3fa446a52be134607b410e6bfa9b7946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2156318ac36bb8e1ee334d298d3f2bff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(192, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c2e87645396ae89fa8c2962d94d36e4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c7e53a772bce50feced9272b08b9b84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 192, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c9681b531783052c0e4d1094737f42e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_92035424d16071704640fceac9cb7ba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d69ec1bcb65db19225d44841159e54d2
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_802fe6cf4c2dbdee0f05c2b952759981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c9681b531783052c0e4d1094737f42e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_012a6fae748b97b18b09071037379250(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77c05b27af938dada5589a07edf0f3e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c9681b531783052c0e4d1094737f42e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bcf3aa8f2195df151329a6e16ce74754(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d1afde021d493e1bfb21897979dab25d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f51da3aa8ea72907a3b1130e603363d0
        def get_inputs(self):
            return [
                paddle.to_tensor([False, True, False, False, False, False], dtype='bool').reshape([6]),
            ]


    class TestPrimitiveOp_ad7ab7da8ed2ead67d8fde22b8a0a257(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f51da3aa8ea72907a3b1130e603363d0
        def get_inputs(self):
            return [
                paddle.to_tensor([False, False, False, False, False, False], dtype='bool').reshape([6]),
            ]


    class TestPrimitiveOp_32904982dc10676823c0da88bc0a8ed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5b801c6443b883acfc88853e629b6b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_16a6891af27d79c6a00c1078c9671f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c6ab8826c46f53673d2600abe55eefe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_09c5b9bec02b10434861ecce5361d95f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 76], dtype='int32'),
            ]


    class TestPrimitiveOp_a10a866c602a2060205e53d11bfae048(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4d325e19128a63c2f05ee741ce186c0
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1366182208f27e283a06776cf6c0d3b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcdb8b145ad4f0d69dd06e27cbec8065
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1759, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_432c5e0b6d7efbef2aa80d7048f323dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e28f58273fde8f46035cfa4d5152c9f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_287c75ca0d09ad51acfc460554a3ff1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e28f58273fde8f46035cfa4d5152c9f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_802fe6cf4c2dbdee0f05c2b952759981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0b0edc8276c10fc9d7bb0e027ab42527(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(28, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_de07304dbd124756772af50e24fac360(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(50, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4ef536d038f8c4fbedd7c2cb6605bec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_182239e1ffe3ed9bf865d9aa1f21632e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4116, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2d92b41333621cbb8087ed27db437c55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a98986c2c7d59eca8b564df20faf43bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7ae9b6fbb50900fca3935271b8f9c384(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[80], dtype='int64'),
            ]


    class TestPrimitiveOp_e5ad3e462b927043dfba2bc14690de27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
            ]


    class TestPrimitiveOp_f16f3f6db598aeedc9ac2faa370ef788(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_8b77bef5be283ff6a7e0bc3996048e65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77938fc03a986310c6c2dddb16ea348e
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b77bef5be283ff6a7e0bc3996048e65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77938fc03a986310c6c2dddb16ea348e
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_17597903c2690c89e3c7a922144153f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc7fa7555f78fce26e42a60aaac74ba5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17597903c2690c89e3c7a922144153f0
        def get_inputs(self):
            return [
                paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7b468f77da28e22ebb426950d328acfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895e28dbe52d7a21dec773cd0e0db85a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3306543529033661, 0.3834773600101471, 0.49611005187034607, 0.11812765151262283, 0.478644460439682, 0.32757896184921265, 0.3514584004878998, 0.3204523026943207, 0.48325031995773315, 0.25242266058921814, 0.4594441056251526, 0.10603760182857513, 0.09884601831436157, 0.039876293390989304, 0.12443646788597107, 0.20525671541690826, 0.031096970662474632, 0.08158884942531586, 0.2653093934059143, 0.3785844147205353, 0.4728078544139862, 0.47888636589050293, 0.37857428193092346, 0.030704200267791748], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_773fd4d7fce0f317f5804c52e34c595f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
            ]


    class TestPrimitiveOp_92bca5d268bb244f177c1c068a51dcc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
            ]


    
    class PrimitiveOp_0bfbf52475df62c2bd9ca324c18284cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.cast(input_0, paddle.float64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ef8f56b59d8638f3b4e4343ac7b49f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bfbf52475df62c2bd9ca324c18284cf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6939955949783325], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_0caf4225bb7d43f2f45a1fa28f10b6b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_2be5d6458dd1cacb5194c5338c64b953(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb94b28ee4f98c79d4c881970a09b471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_c4b3c17c2c1522493c2c36d22ceb6535(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9e89e028c6d753fa4b1735f1872f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c92e0ec02de5625fd55b34449d684532(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6069, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_25805ace3e67d7710cb359976372fb7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b922e0140815222aeead37277d6a6c33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_743a5a03ac06a52ed53109171771bc25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c6ab8826c46f53673d2600abe55eefe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b641c3c6ab96385f5dab4ac7fee966e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_6ca2a2d6c9e2cc5266de48a08bebc0dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4d325e19128a63c2f05ee741ce186c0
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2294baf6a9045885efa07f50617bd72d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcdb8b145ad4f0d69dd06e27cbec8065
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1538, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_32904982dc10676823c0da88bc0a8ed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_32904982dc10676823c0da88bc0a8ed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_fb24466f7200249d72d073ecd43bb063(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e47b6d269e54aa74ea62513a14d6489
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    class TestPrimitiveOp_d3a6946c78b24aa144d7f6d07060289b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17597903c2690c89e3c7a922144153f0
        def get_inputs(self):
            return [
                paddle.to_tensor([8, 2], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_77d8e041bbee4adad3a29fc4bea8b00f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895e28dbe52d7a21dec773cd0e0db85a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.004606406204402447, 0.23123939335346222, 0.4518365263938904, 0.139329195022583], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_ed70912a87c6e9b766ec49dc5a1bced8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_61a5b77fcd93a97e5f2607de33269744(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_904e9b9355d7e13194363e52c0f2f695(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(52, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b5489bc08cb69df0cd70fb6389691632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(202, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_bb8d3995b3b5a551c656ee8b23ee0c42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1025, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_796b607a4382e0b10b6ff907669d98cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype='int64').reshape([14]),
            ]


    class TestPrimitiveOp_056fbea28a02c2047238167f4d976322(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([14, 14, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_454a6d1d67341954e0eb8d24bbe0be61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([14, 14, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82d2c076e36498f2e9ae2258a5f0ef08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], dtype='int64').reshape([28]),
            ]


    class TestPrimitiveOp_033d45f1adcea6c23a2520b9b5c691ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([28, 28, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a18abcbf10ea866b2ccd105bc21ec448(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([28, 28, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c4694e2a405974fdfad12fac3028cab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[56], dtype='int64'),
            ]


    class TestPrimitiveOp_6a91bbdddcc088c4445856596f175334(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([56, 56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d5800550c301415f5b77e8b8a13acee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([56, 56, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_60122ffd4eb2fc5ee94a65f12f7cf2e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_60122ffd4eb2fc5ee94a65f12f7cf2e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_de5243416defd095a8f4901acb914079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(104, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_de5243416defd095a8f4901acb914079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(104, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_069aabc70ea7b88a2560e3bc2d503ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_069aabc70ea7b88a2560e3bc2d503ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7be783bb5c395cadf9b50bba9f22adf9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e47b6d269e54aa74ea62513a14d6489
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10cdb2ae7d8bba2fdab34a87674d6a63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e84ab84058793cee2719f0ee499707f4
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5be6741a1a32bd81e70e6152de3c2ebf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e84ab84058793cee2719f0ee499707f4
        def get_inputs(self):
            return [
                paddle.to_tensor(7, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_489fd001f1dce121f475610cc66041e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcf1f3382b59bc773303e88b5072340e
        def get_inputs(self):
            return [
                paddle.to_tensor([300.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cec54815ac05282cf7f41838d06b6482(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f4ef536d038f8c4fbedd7c2cb6605bec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_069aabc70ea7b88a2560e3bc2d503ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_31fa21a12a2ee949a57731dde428da9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_34d013d5f0f7c0aa4fc14b5696c6b26d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c6ab8826c46f53673d2600abe55eefe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_83893f1f2d9cd56a209264401453d93b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_ef67089c6caa62765048441517ddf209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4d325e19128a63c2f05ee741ce186c0
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d55eb051833fbf610a582b3557f44671(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcdb8b145ad4f0d69dd06e27cbec8065
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2135, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_1f617b341f517a443d6c09a453fc7f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(14, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8d063c09f60fcc9258d34e3e9b863cd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(25, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f8d26dc573bfd532cf6be4dab502c17c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_87b7427dbd8306c8896975cd7d98e496(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_060e9e9c44e7e984ab239ef0d23ae79c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c6ab8826c46f53673d2600abe55eefe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_e6b597715457a7207fd307c04e43470f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_a5fe860a811350e31c1bd6ee54ef3acb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4d325e19128a63c2f05ee741ce186c0
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1fac90b82dc6b92cf9e1c2f064f15d9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcdb8b145ad4f0d69dd06e27cbec8065
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4590, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_8a2c86d3a221afc1c9dfefd13e298794(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32a19618a8f19392b62d9a801d8cac68
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[6, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_4b1d2e5405b199bdf798aad70af5c9ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ac4ca313e7fbd3c10e159cd7bd7c0be
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_67046a6efbf3a7a1bd2ad9436cd0f103(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_20a00d63dfd16aaffd1c54590e323f87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_d631209889d48437bb88d7114965cfcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c6ab8826c46f53673d2600abe55eefe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9410f8524fa44a445a7da6bc8a5945f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_4c9fc59ef981fec0bb41c7f728fc6106(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4d325e19128a63c2f05ee741ce186c0
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b5f443288ede3be83ea2a02aadff1c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcdb8b145ad4f0d69dd06e27cbec8065
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1042, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_159853c5be6e7ff71d2a2861bf97f1ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(9261, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bcf3aa8f2195df151329a6e16ce74754(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8c501a97daf9a65ee9fd09698c997571(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[68], dtype='int64'),
            ]


    class TestPrimitiveOp_53b48b3d62e4f9e15b0db3614cc7d3d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[34], dtype='int64'),
            ]


    class TestPrimitiveOp_3d673d6367a9466d34dc7c89ec8688ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype='int64').reshape([17]),
            ]


    class TestPrimitiveOp_4df91d36cbe4ee7afcc2b5fb509be472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77938fc03a986310c6c2dddb16ea348e
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4df91d36cbe4ee7afcc2b5fb509be472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77938fc03a986310c6c2dddb16ea348e
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1f29ca91e7986af780fd3cc740eda9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2257ef4974735a200c547013f17cd828(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1f29ca91e7986af780fd3cc740eda9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2257ef4974735a200c547013f17cd828(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1f29ca91e7986af780fd3cc740eda9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2257ef4974735a200c547013f17cd828(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_eb531f5775a9f575efca9cb6a04c768b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(2048, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_43306b382dae4416341ddd59242796f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd15b377ada49352fc11600cd0df44b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2048, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_c4b3c17c2c1522493c2c36d22ceb6535(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9e89e028c6d753fa4b1735f1872f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_2be5d6458dd1cacb5194c5338c64b953(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb94b28ee4f98c79d4c881970a09b471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_0caf4225bb7d43f2f45a1fa28f10b6b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_617dab664ac3ba7b80a678548a4279e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_617dab664ac3ba7b80a678548a4279e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_54070f8b3c1d480cc016f2361e482ff6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([16, 16, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2d1f7851632fad7b8414226f4a5d21ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([8]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2d1f7851632fad7b8414226f4a5d21ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([8]),
            ]


    class TestPrimitiveOp_eec36f0f6b6ea5bf4dd3fcea50d52365(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([8, 8, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e244c47b82deaf9f39c116ef0e74df1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(2100, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0824d25a8f6e3222b14dfdcb8c1348de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2257ef4974735a200c547013f17cd828(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0824d25a8f6e3222b14dfdcb8c1348de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2257ef4974735a200c547013f17cd828(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0824d25a8f6e3222b14dfdcb8c1348de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2257ef4974735a200c547013f17cd828(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_eb531f5775a9f575efca9cb6a04c768b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(2048, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6aee5b00238ca8faed8a59e883b06b7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd15b377ada49352fc11600cd0df44b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2048, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8d3995b3b5a551c656ee8b23ee0c42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1025, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b51694e6ab84e392d6f6ac9902de4f15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_79cbecb3602c323611c9c681049e2d5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_293f89ffa476e87db6fdd7e16d0bb986(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c6ab8826c46f53673d2600abe55eefe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_41c5a2fb278cc53ec619c9bff6408214(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_b8195cf1180f33184c0a76e9205a746c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4d325e19128a63c2f05ee741ce186c0
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bba4cee86e480c52ddba43dc3d0281f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcdb8b145ad4f0d69dd06e27cbec8065
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2339, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_802fe6cf4c2dbdee0f05c2b952759981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ec485f2c5df474efa49cb9604a0c6325(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_4010f8a48063ccdf657a4792ee9d9998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_0e9b7ec58d801cd7b2a23b59e663ab5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c6ab8826c46f53673d2600abe55eefe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_3749d678a6e865ae5df22f3463afb3cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_7c7f57f4cd66d8f9001c09835fbef201(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4d325e19128a63c2f05ee741ce186c0
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ec81065481b936fbe216fa3973e24e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcdb8b145ad4f0d69dd06e27cbec8065
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3063, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_980c04af2955f51824b4abca7a6deb38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_02c6cddc7b23e158a6e4f6e497956af0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_6653c48e2d4ad1598423c84dfa76668f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c6ab8826c46f53673d2600abe55eefe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f10f2d09f9e7352f13757a7e363c76da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_1b39147b688a1761f6ad3f12b26c5bb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4d325e19128a63c2f05ee741ce186c0
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78b2d7fa18050bf4bf096663742344f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcdb8b145ad4f0d69dd06e27cbec8065
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3822, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5941bc0de4eba289d0e5be4b4f2214b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcf1f3382b59bc773303e88b5072340e
        def get_inputs(self):
            return [
                paddle.to_tensor([100.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_cca72d2a6439ce40b2a2d3b7b6c0a4d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11109, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f65445d4f28e8755b69d5275765232b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_63fb5f540d40add592fdf67694c9eea9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32a19618a8f19392b62d9a801d8cac68
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_b52ae596a224b3b798f54c1fa8fcfc02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d69ec1bcb65db19225d44841159e54d2
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_13beb322da3061bca6147d41a530abd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d69ec1bcb65db19225d44841159e54d2
        def get_inputs(self):
            return [
                paddle.to_tensor([11], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_321c844fff06588e1bf0c3ff75d364f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d69ec1bcb65db19225d44841159e54d2
        def get_inputs(self):
            return [
                paddle.to_tensor([384], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d19da8a314e7b12538cf533336b63bf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d69ec1bcb65db19225d44841159e54d2
        def get_inputs(self):
            return [
                paddle.to_tensor([28], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9e98cb495112f97b9d63108c94204307(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d69ec1bcb65db19225d44841159e54d2
        def get_inputs(self):
            return [
                paddle.to_tensor([77], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_135e543d89695d997c3bcd4989fcc795(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[152], dtype='int64'),
            ]


    class TestPrimitiveOp_57cf7baf072d859fd064411b6bf6c9e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[100], dtype='int64'),
            ]


    class TestPrimitiveOp_00c5bdfa44e77baa21ef5357d7f4d9de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([100, 152, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_955a6ccf34f1de3c76031abbb94b2395(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([100, 152, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_370ed95f7029bad2d8c319b2f72c8f1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[76], dtype='int64'),
            ]


    class TestPrimitiveOp_4543bf32baae456bf6ebd7be45836297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[50], dtype='int64'),
            ]


    class TestPrimitiveOp_61bccb35f5075a20b12823674039bf86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([50, 76, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5bccab51d64fd39b5458ea7ce4433099(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([50, 76, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f91dbb08eaf8765b074c49f3bc4231d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[38], dtype='int64'),
            ]


    class TestPrimitiveOp_3f1ffbb06b5c83836774f012484008e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype='int64').reshape([25]),
            ]


    class TestPrimitiveOp_98d38f21a3911e179744050f1b4c2213(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([25, 38, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c013b1faa080f13ac681fa22e18d0d02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([25, 38, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7370a972d9492fdf9e2b6421c7ffa9b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype='int64').reshape([19]),
            ]


    class TestPrimitiveOp_69e21721177d8647451f4fed7881c4b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype='int64').reshape([13]),
            ]


    class TestPrimitiveOp_8f6e5a20f52a4a89feddea63f8b362b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([13, 19, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_849adb3e083b64046b8aa8b6104eb5b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([13, 19, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad2e0866abea2cdd2d2964258a39d89b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64').reshape([10]),
            ]


    class TestPrimitiveOp_8ce205fef50d91d11cf65ae98990af23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6], dtype='int64').reshape([7]),
            ]


    class TestPrimitiveOp_fbe13c07644080e2e69baa57d878b28b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([7, 10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c11a2a311b95583cf1af341648b4f0a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([7, 10, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_62ce5c63d2802e3b608db68d14cdadc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a98986c2c7d59eca8b564df20faf43bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_72fb9d711e4dbc2aa93336fb8a49a4ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e28f58273fde8f46035cfa4d5152c9f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c9681b531783052c0e4d1094737f42e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d1ba26cfb1b248a545fe9250efbb97e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895e28dbe52d7a21dec773cd0e0db85a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.34141385555267334, 0.08070738613605499, 0.03860168904066086, 0.49608147144317627, 0.41707998514175415, 0.07405710965394974, 0.39049777388572693, 0.14312376081943512, 0.4575801193714142, 0.44072431325912476, 0.48256993293762207, 0.10408809781074524, 0.3957173824310303, 0.045056845992803574, 0.2210846096277237, 0.11286043375730515, 0.109720878303051, 0.17332060635089874, 0.47122129797935486, 0.48317795991897583], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_91c1e9c4d62c453dff72e86b2b60229f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_d0aea1398949b006b5e02c7358179872(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_0caf4225bb7d43f2f45a1fa28f10b6b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_2be5d6458dd1cacb5194c5338c64b953(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb94b28ee4f98c79d4c881970a09b471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_c4b3c17c2c1522493c2c36d22ceb6535(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9e89e028c6d753fa4b1735f1872f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_cec54815ac05282cf7f41838d06b6482(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f4e206aed9ba60a531d4cab6820b4e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_069aabc70ea7b88a2560e3bc2d503ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_31fa21a12a2ee949a57731dde428da9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_34d013d5f0f7c0aa4fc14b5696c6b26d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c6ab8826c46f53673d2600abe55eefe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_83893f1f2d9cd56a209264401453d93b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_2ddd44d76c3f1e709f3ca3720819a1d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4d325e19128a63c2f05ee741ce186c0
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_479ca43f21e534fae394d8996531578b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcdb8b145ad4f0d69dd06e27cbec8065
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2057, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_026d73ec489d54530b281f2d73e7ab94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2257ef4974735a200c547013f17cd828(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5174cf0d2e3ef2d1885f95a0fad6fb3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10537b3e46df3960a74b35271bb061b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0af5f14fbb5c825ceafed43ed9515b50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(3024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ef1ccc76a8035835ee9d8a5de828c7d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_4f9f915e6e926b1782ed252222fac7f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[72], dtype='int64'),
            ]


    class TestPrimitiveOp_3fa446a52be134607b410e6bfa9b7946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_9c1e956805104a4bbd0ecf39b7bf1c04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype='int64').reshape([18]),
            ]


    class TestPrimitiveOp_b000c3f56ff28e745270f43886349128(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77938fc03a986310c6c2dddb16ea348e
        def get_inputs(self):
            return [
                paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b000c3f56ff28e745270f43886349128(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77938fc03a986310c6c2dddb16ea348e
        def get_inputs(self):
            return [
                paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_ea0df210b00f923541bf6bcc99305f0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1174, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e8df93d5e317d77438ed6c60e8173dbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d69ec1bcb65db19225d44841159e54d2
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_730322fdc76616c6ac52fcabfe2d2bf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0824d25a8f6e3222b14dfdcb8c1348de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2257ef4974735a200c547013f17cd828(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_cec54815ac05282cf7f41838d06b6482(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_239058c1469b86d14e0f7bda6131b249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_918e26562114cbe1be4dafd5e846bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2d92b41333621cbb8087ed27db437c55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d584f3671d10c8df4d8cbab851020f
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a98986c2c7d59eca8b564df20faf43bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc3d010e6808621cdbaa88a7eb909536
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156290f824910d1f44bbd7b0d7c5f5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5d998325673a24e20286d59698f863bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ade39e9117dc96145d7ae1006aab434
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_40c7bd48ddecc2ef19d81f11e48ba6ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_23d05570eb7c560c79301f62d5a87d3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c6ab8826c46f53673d2600abe55eefe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a02b59a31439a109f7b9154823b2ec0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2daf7d5fceb3526b254e00d16ceaa8c2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_c4a31aa87d9c59583e9147addaa4b2c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4d325e19128a63c2f05ee741ce186c0
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a52e29ce60fe8571f0b957ea9d07012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcdb8b145ad4f0d69dd06e27cbec8065
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4189, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_ea0df210b00f923541bf6bcc99305f0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1174, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9453659f9e4fe376b8186e33cbcd7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9664fa274b61deaa1f0ee0d55d697428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2694f0c7f667b1479a04c20f6961e367
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_8076e462c5b8d0dbcea41907d19fc46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2f2b5cbfd9d587d4e9f3299413ded0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_0caf4225bb7d43f2f45a1fa28f10b6b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_194d5172b4ad35e5a990636226a017e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_4c5e1b1a2d3647042218f9c004baf78e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a0caca3fc71a6c28ca03883c9e3cd63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_2be5d6458dd1cacb5194c5338c64b953(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb94b28ee4f98c79d4c881970a09b471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_7c98aa9d63abd926d5bbdd6eefb3a32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_bb8e8a81f68250cc8559d270cfdb16cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7915592b935e4847aaf184d0b09e7b57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_c4b3c17c2c1522493c2c36d22ceb6535(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc31cf69c43c50455aa70840dd6ab54b
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9e89e028c6d753fa4b1735f1872f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09a0373c8051d470e4227f441acc8d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b424a347d9b0849cec6584fe74e3075f
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    

if __name__ == '__main__':
    unittest.main()