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
    class PrimitiveOp_0c062d85c46fb003eeedcf8508892afc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.floor_divide(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4fde453e52c11bb304df51cce7560d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(528, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_18f6a64b1708929c16c8db90681fce54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(12, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_53648090609638bfa35e7ee75c334dd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(384, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_700e4f900808808a5441bcab5e8baac3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(20, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_df8f724edc6a32d779f9de2c5b033fc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(2, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ec98876fd863f6e909c4a63ba835873a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d99857c27f1a71319bad4e456c4cecc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(576, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_95bec1b5ec8dd4deca95f64f1c39e689(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(96, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_18f6a64b1708929c16c8db90681fce54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(12, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0fa404ab43430cd619b097efb40efa47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(960, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c2ad8087f7a8aeee70ac52e34c473d5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(2112, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_4c271cfa78e43acb409f91d1bbc4ee13(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.floor_divide(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06cda83e35e44042ed4f3cbc99976b7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c271cfa78e43acb409f91d1bbc4ee13
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int32').reshape([1]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7685f8c721afbbdb9875f2b234051b5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c271cfa78e43acb409f91d1bbc4ee13
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4f8b06a4ba9f270a619b2d418133722d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5760f53b1d4f387710da3889a2e299a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(240, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0aea87cc60def9d3f831b6d44233977f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(44, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_dc2eba7d7ddf3bb1ff6127d5ef7f655a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c271cfa78e43acb409f91d1bbc4ee13
        def get_inputs(self):
            return [
                paddle.to_tensor([28], dtype='int32').reshape([1]),
                paddle.to_tensor(7, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a15ae69d98c4f4d1f3d2ba66a066da63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c271cfa78e43acb409f91d1bbc4ee13
        def get_inputs(self):
            return [
                paddle.to_tensor([77], dtype='int32').reshape([1]),
                paddle.to_tensor(7, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_67c67a1a1da150fb49873368b9f2814a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(144, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4fde453e52c11bb304df51cce7560d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(528, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_18f6a64b1708929c16c8db90681fce54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(12, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_53648090609638bfa35e7ee75c334dd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(384, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_700e4f900808808a5441bcab5e8baac3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(20, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_df8f724edc6a32d779f9de2c5b033fc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(2, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ec98876fd863f6e909c4a63ba835873a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d99857c27f1a71319bad4e456c4cecc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(576, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_95bec1b5ec8dd4deca95f64f1c39e689(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(96, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_18f6a64b1708929c16c8db90681fce54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(12, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0fa404ab43430cd619b097efb40efa47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(960, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c2ad8087f7a8aeee70ac52e34c473d5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(2112, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_06cda83e35e44042ed4f3cbc99976b7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c271cfa78e43acb409f91d1bbc4ee13
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int32').reshape([1]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7685f8c721afbbdb9875f2b234051b5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c271cfa78e43acb409f91d1bbc4ee13
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4f8b06a4ba9f270a619b2d418133722d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5760f53b1d4f387710da3889a2e299a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(240, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0aea87cc60def9d3f831b6d44233977f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(44, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_dc2eba7d7ddf3bb1ff6127d5ef7f655a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c271cfa78e43acb409f91d1bbc4ee13
        def get_inputs(self):
            return [
                paddle.to_tensor([28], dtype='int32').reshape([1]),
                paddle.to_tensor(7, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a15ae69d98c4f4d1f3d2ba66a066da63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c271cfa78e43acb409f91d1bbc4ee13
        def get_inputs(self):
            return [
                paddle.to_tensor([77], dtype='int32').reshape([1]),
                paddle.to_tensor(7, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_67c67a1a1da150fb49873368b9f2814a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(144, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4fde453e52c11bb304df51cce7560d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(528, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_18f6a64b1708929c16c8db90681fce54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(12, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_53648090609638bfa35e7ee75c334dd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(384, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_700e4f900808808a5441bcab5e8baac3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(20, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_df8f724edc6a32d779f9de2c5b033fc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(2, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ec98876fd863f6e909c4a63ba835873a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d99857c27f1a71319bad4e456c4cecc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(576, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_95bec1b5ec8dd4deca95f64f1c39e689(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(96, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_18f6a64b1708929c16c8db90681fce54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(12, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0fa404ab43430cd619b097efb40efa47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(960, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c2ad8087f7a8aeee70ac52e34c473d5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(2112, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_aed4a1eee7469875cb9ebbee2418bef6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.floor_divide(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_81a493e8bcddea9720ab4698a5f1a040(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed4a1eee7469875cb9ebbee2418bef6
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int32').reshape([1]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9ea164e7ce51fcd83f3c2e0326f991c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed4a1eee7469875cb9ebbee2418bef6
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4f8b06a4ba9f270a619b2d418133722d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5760f53b1d4f387710da3889a2e299a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(240, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0aea87cc60def9d3f831b6d44233977f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(44, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cdd63a3a5c085d6768d0958e69784bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed4a1eee7469875cb9ebbee2418bef6
        def get_inputs(self):
            return [
                paddle.to_tensor([28], dtype='int32').reshape([1]),
                paddle.to_tensor(7, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1f6264669acddd467a9c2764d281ea03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed4a1eee7469875cb9ebbee2418bef6
        def get_inputs(self):
            return [
                paddle.to_tensor([77], dtype='int32').reshape([1]),
                paddle.to_tensor(7, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_67c67a1a1da150fb49873368b9f2814a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c062d85c46fb003eeedcf8508892afc
        def get_inputs(self):
            return [
                paddle.to_tensor(144, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    

if __name__ == '__main__':
    unittest.main()