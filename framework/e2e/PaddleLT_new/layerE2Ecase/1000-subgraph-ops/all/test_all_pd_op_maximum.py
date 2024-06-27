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
    class PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_86c4c3277305dd6ddff1af0f5290982f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86c4c3277305dd6ddff1af0f5290982f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3dad9b4bf6a3008eb6f309b0341e9227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3dad9b4bf6a3008eb6f309b0341e9227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8468c8c23b232dd0cb87269ce1f2b387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8468c8c23b232dd0cb87269ce1f2b387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d67af0770c1a44070f76e873e81e675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d67af0770c1a44070f76e873e81e675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a86f8642b02d880fcb8a4a380e88ab00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a86f8642b02d880fcb8a4a380e88ab00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d90e35866eba82b9904def88da3cb1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d90e35866eba82b9904def88da3cb1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86c4c3277305dd6ddff1af0f5290982f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86c4c3277305dd6ddff1af0f5290982f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9630ce511e3c8b9c83e9bf9784841f8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9630ce511e3c8b9c83e9bf9784841f8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0dd44a76c7a4e8447bf424768eadb13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0dd44a76c7a4e8447bf424768eadb13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_19d67d9681bf764c1f1af2fcb3b993ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19d67d9681bf764c1f1af2fcb3b993ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19d67d9681bf764c1f1af2fcb3b993ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19d67d9681bf764c1f1af2fcb3b993ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc1d3bdade8d147f6188485410ffdb17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc1d3bdade8d147f6188485410ffdb17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f27bc3cf81be12de4e3e0f186b99f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f27bc3cf81be12de4e3e0f186b99f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e27a01469fb2ab3e752bb1143a535d1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e27a01469fb2ab3e752bb1143a535d1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9e6a88155731b4eda272425ecf74d32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9e6a88155731b4eda272425ecf74d32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_88a5e4090695dc5292a01265289865cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2afbed2592e5cc4a5551e79e22801598(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33700039982795715], [0.13599033653736115], [0.16089512407779694], [0.4550243616104126], [0.2522077262401581], [0.41581976413726807], [0.06474526226520538], [0.22361089289188385], [0.08131606876850128]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.31189051270484924], [0.1544954627752304], [0.3554002642631531], [0.06358063966035843], [0.38996535539627075], [0.37870699167251587], [0.02283250354230404], [0.03404628112912178], [0.4961751103401184]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_60024d8db5bc3243b383d9aa54b39b5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17808885872364044], [0.1587141752243042], [0.2988268733024597], [0.33936357498168945], [0.37306493520736694], [0.3378402590751648], [0.11509011685848236], [0.29118287563323975], [0.28715723752975464]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.08929294347763062], [0.13941359519958496], [0.3334523141384125], [0.2755478620529175], [0.273370623588562], [0.07012606412172318], [0.0644788146018982], [0.1514199823141098], [0.44735172390937805]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_d035f9ebefd35d55b2b35a1b09073b3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40283963084220886], [0.45022642612457275], [0.4985415041446686], [0.32824042439460754], [0.11407893896102905], [0.04877207800745964], [0.24639740586280823], [0.4660728871822357], [0.06258141249418259]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.10231184214353561], [0.21500952541828156], [0.033458199352025986], [0.3819327652454376], [0.053689487278461456], [0.17333781719207764], [0.3545892834663391], [0.06132432818412781], [0.08616641163825989]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_dbafd909fd12b0df2209061f14c98568(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07969757914543152], [0.14082278311252594], [0.477491170167923], [0.15222904086112976], [0.2556220293045044], [0.2010723203420639], [0.2752452790737152], [0.257482647895813], [0.3754969537258148]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4645979106426239], [0.151620551943779], [0.23870912194252014], [0.4139777421951294], [0.40845590829849243], [0.21291503310203552], [0.35330283641815186], [0.264354944229126], [0.2503066658973694]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_efb485fff2c50d6ac6cac7b5e63216f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efb485fff2c50d6ac6cac7b5e63216f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efb485fff2c50d6ac6cac7b5e63216f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efb485fff2c50d6ac6cac7b5e63216f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e95089378bd99bd9f812657fb508a008(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f2c7e92d0a23f8980620768017275982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.32048559188842773, 0.04365863651037216, 0.14801615476608276, 0.059532683342695236, 0.1973104625940323, 0.10881417244672775], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c194de1968e31cedec708a7595e5077d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.43007832765579224, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3242815434932709, 0.21491502225399017, 0.4360129237174988, 0.2119390070438385, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_a1613e2d37c0b30331bd3db47a459e46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4498527944087982, 0.4890660345554352, 0.20852655172348022, 0.4761844575405121, 0.404381662607193, 0.10783115774393082], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_16cf1af4cbe311e51bdf562f63b55af4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.43007832765579224, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1643696129322052, 0.4882410168647766, 0.43661245703697205, 0.06498825550079346, 0.49191561341285706, 0.33490511775016785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_fed2be89f18f829301a603fe2323a009(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.45263877511024475, 0.18129011988639832, 0.45884010195732117, 0.49139586091041565, 0.3470446765422821, 0.2975061535835266], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8b52a3427a5a405c47e167e6f3f1e18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.39286479353904724, 0.4360129237174988, 0.43007832765579224, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3685716390609741, 0.420699805021286, 0.35412517189979553, 0.2729951739311218, 0.31833115220069885, 0.4968334138393402], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_718141fbc1bd5abfe1617a3cc9a0c63a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_718141fbc1bd5abfe1617a3cc9a0c63a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_718141fbc1bd5abfe1617a3cc9a0c63a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_718141fbc1bd5abfe1617a3cc9a0c63a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_15306c9635961599f60a1355bcd06695(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3ab74ebddd2598567dec017e0a6e27ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15306c9635961599f60a1355bcd06695
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1.000000013351432e-10], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3d67af0770c1a44070f76e873e81e675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d67af0770c1a44070f76e873e81e675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f27bc3cf81be12de4e3e0f186b99f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f27bc3cf81be12de4e3e0f186b99f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18ed0a638cf295cf2c42a98aee4e3c53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18ed0a638cf295cf2c42a98aee4e3c53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18ed0a638cf295cf2c42a98aee4e3c53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18ed0a638cf295cf2c42a98aee4e3c53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d90e35866eba82b9904def88da3cb1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d90e35866eba82b9904def88da3cb1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3dad9b4bf6a3008eb6f309b0341e9227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3dad9b4bf6a3008eb6f309b0341e9227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f4aeb903741b82161ab397e2c158a34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.24695035815238953]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.1818336695432663]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_1c536259d63dbe2cc5f67ff2c18847df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.27610716223716736]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.028087755665183067]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_85643ea3b519a973808a08713d188cfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20727156102657318]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.44695910811424255]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ca6c191223432795bb978e2d9746b9be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.35763055086135864]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3456944525241852]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_43bb972d96de9dd14ce449c0af2a7105(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06345830112695694], [0.37558823823928833], [0.1739027202129364], [0.027879230678081512], [0.49161165952682495], [0.11910757422447205]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17218370735645294], [0.19266913831233978], [0.3696064352989197], [0.3602122664451599], [0.343851238489151], [0.4164700210094452]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_9781246a6316269b6468672fc48f0842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42717310786247253], [0.2651657462120056], [0.46184709668159485], [0.13699816167354584], [0.30248284339904785], [0.26277780532836914]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.01432943344116211], [0.07210607081651688], [0.11303866654634476], [0.4184499979019165], [0.4230078458786011], [0.2603096663951874]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_2e6a4aad420411bbd3accbd1870de0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1136389821767807], [0.3263870179653168], [0.4104270935058594], [0.2049286961555481], [0.05052289366722107], [0.18513862788677216]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06397107243537903], [0.32089540362358093], [0.02683880925178528], [0.38781505823135376], [0.12123280763626099], [0.1445770114660263]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_4dd7d3b9a622587d2bfdc56bf3907fdc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3679124414920807], [0.3236524164676666], [0.1917470246553421], [0.0763665959239006], [0.3653966784477234], [0.06940227746963501]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.07160773128271103], [0.0910111665725708], [0.3088380694389343], [0.12214173376560211], [0.39863425493240356], [0.18909907341003418]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_bc1d3bdade8d147f6188485410ffdb17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc1d3bdade8d147f6188485410ffdb17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8468c8c23b232dd0cb87269ce1f2b387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8468c8c23b232dd0cb87269ce1f2b387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9d3b7dd15a2ecddaca2182432a86de9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9d3b7dd15a2ecddaca2182432a86de9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9d3b7dd15a2ecddaca2182432a86de9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9d3b7dd15a2ecddaca2182432a86de9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e95c45b5ce0c8bce6a4c17a7df65c4d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e95c45b5ce0c8bce6a4c17a7df65c4d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5553f4e6ee0edaacff0ba444daceecc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5553f4e6ee0edaacff0ba444daceecc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5553f4e6ee0edaacff0ba444daceecc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5553f4e6ee0edaacff0ba444daceecc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b769a30c8b00f20ea1b737b3c5f8cd70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b769a30c8b00f20ea1b737b3c5f8cd70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b769a30c8b00f20ea1b737b3c5f8cd70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b769a30c8b00f20ea1b737b3c5f8cd70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a86f8642b02d880fcb8a4a380e88ab00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a86f8642b02d880fcb8a4a380e88ab00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e27a01469fb2ab3e752bb1143a535d1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e27a01469fb2ab3e752bb1143a535d1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fae8c5c800da71f1f6c0feab207b4a4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29145920276641846], [0.07165557891130447], [0.07205324620008469], [0.22017629444599152], [0.1261482983827591]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3746805787086487], [0.2069297879934311], [0.08722388744354248], [0.3281881809234619], [0.10755358636379242]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_ff3cbdc45a27bb5ce2addc2657624b9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.446043998003006], [0.12132035940885544], [0.0887727364897728], [0.17385192215442657], [0.07668683677911758]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2575170397758484], [0.09787295013666153], [0.30022138357162476], [0.40387892723083496], [0.24005642533302307]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_629f68d6d3b3baef4d5c068ae60e5a31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1386314034461975], [0.3368445634841919], [0.4964533746242523], [0.08265111595392227], [0.13507212698459625]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.104369156062603], [0.09522414207458496], [0.19618113338947296], [0.1922953873872757], [0.1039789542555809]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_33e42703e8b2c861a5ed1b60c9e53a17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05733131244778633], [0.1570308655500412], [0.3689388930797577], [0.17223645746707916], [0.2121700644493103]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.4198094606399536], [0.43931514024734497], [0.03426346927881241], [0.41905465722084045], [0.24269208312034607]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_c0dd44a76c7a4e8447bf424768eadb13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0dd44a76c7a4e8447bf424768eadb13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9e6a88155731b4eda272425ecf74d32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9e6a88155731b4eda272425ecf74d32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db140cfb159ed29f63fe09694cb8a42d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db140cfb159ed29f63fe09694cb8a42d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c5329cbe5b7a01d672b92d363b6b6c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c5329cbe5b7a01d672b92d363b6b6c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c5329cbe5b7a01d672b92d363b6b6c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c5329cbe5b7a01d672b92d363b6b6c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da63c2e2a882b71f066802aba4b40a60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da63c2e2a882b71f066802aba4b40a60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da63c2e2a882b71f066802aba4b40a60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da63c2e2a882b71f066802aba4b40a60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_992386ca0fe7d115f0fac805aa067064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_992386ca0fe7d115f0fac805aa067064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_992386ca0fe7d115f0fac805aa067064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_992386ca0fe7d115f0fac805aa067064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e95c45b5ce0c8bce6a4c17a7df65c4d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e95c45b5ce0c8bce6a4c17a7df65c4d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57b08af9e9fe107bfb775ad293c1900b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41876405477523804], [0.0375184640288353], [0.09816955029964447], [0.42315077781677246]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2992156445980072], [0.462017297744751], [0.11372362822294235], [0.295624315738678]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_2b29dc3d7f378c27f9fca38e3d3500d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13186973333358765], [0.09202171117067337], [0.08312810212373734], [0.2566831707954407]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2604764997959137], [0.1259506195783615], [0.3157579004764557], [0.2131945937871933]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_9cf9b730f21ce06af1a5a099f6559338(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10784570127725601], [0.43085747957229614], [0.18588005006313324], [0.29377901554107666]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.4131563901901245], [0.2982136011123657], [0.22558467090129852], [0.03406929969787598]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_a229717e6b9be30be85c21065e9701fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3623265326023102], [0.025396505370736122], [0.3224317133426666], [0.4281119704246521]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.33433759212493896], [0.12095697224140167], [0.2845987379550934], [0.09898775070905685]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_b0f7ebd5135ad81ad04fb7e4482adf85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0f7ebd5135ad81ad04fb7e4482adf85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0f7ebd5135ad81ad04fb7e4482adf85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0f7ebd5135ad81ad04fb7e4482adf85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db140cfb159ed29f63fe09694cb8a42d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db140cfb159ed29f63fe09694cb8a42d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9630ce511e3c8b9c83e9bf9784841f8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9630ce511e3c8b9c83e9bf9784841f8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a1cc7361323b317054a26dd43f98f92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a1cc7361323b317054a26dd43f98f92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3794829ec061c463d674074cc542449c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3794829ec061c463d674074cc542449c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3794829ec061c463d674074cc542449c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3794829ec061c463d674074cc542449c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a1cc7361323b317054a26dd43f98f92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a1cc7361323b317054a26dd43f98f92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a3e97e3483eecd9cbef3c4cb12adc5ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d17de3434724ed1ba94d9112cb16a29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e97e3483eecd9cbef3c4cb12adc5ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d17de3434724ed1ba94d9112cb16a29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e97e3483eecd9cbef3c4cb12adc5ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cb3320a5f060ad2659624ce4621d9127(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4600577d72a8aa2f1d6d4ca4da4f7532(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb3320a5f060ad2659624ce4621d9127
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4600577d72a8aa2f1d6d4ca4da4f7532(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb3320a5f060ad2659624ce4621d9127
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e2033d6afb68542ddf01e093b04f571b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_43c2ad80706215cf0f77acaf4ff89dcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2033d6afb68542ddf01e093b04f571b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43c2ad80706215cf0f77acaf4ff89dcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2033d6afb68542ddf01e093b04f571b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2725aac8670ddbd93636903174f8262f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_88bb50f5a129999fbe0431b50f0c16da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2725aac8670ddbd93636903174f8262f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88bb50f5a129999fbe0431b50f0c16da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2725aac8670ddbd93636903174f8262f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_764eeff18c858cdafc393ac6835aa53f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ad01110cfbf8606f3a39c890c444c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_764eeff18c858cdafc393ac6835aa53f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ad01110cfbf8606f3a39c890c444c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_764eeff18c858cdafc393ac6835aa53f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9601160761ea3127d8587b22969d6fea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_47e450c0ceab4c338e7791427d80cc03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9601160761ea3127d8587b22969d6fea
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47e450c0ceab4c338e7791427d80cc03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9601160761ea3127d8587b22969d6fea
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d17de3434724ed1ba94d9112cb16a29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e97e3483eecd9cbef3c4cb12adc5ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d17de3434724ed1ba94d9112cb16a29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e97e3483eecd9cbef3c4cb12adc5ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4dbf42f4978cca69a91ba2f31b774922(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_db7b3fa80cbf7b98dd5936a4f1a859e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dbf42f4978cca69a91ba2f31b774922
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db7b3fa80cbf7b98dd5936a4f1a859e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dbf42f4978cca69a91ba2f31b774922
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e3578a466474c5a817b5a6b3631fb846(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c85083743241021347f8171e2fbe23a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3578a466474c5a817b5a6b3631fb846
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c85083743241021347f8171e2fbe23a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3578a466474c5a817b5a6b3631fb846
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b2017d3bc8a8646cdfa817f0a89c1389(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1696, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1696, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1a1cb919d48a1c62aa894739651ca55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2017d3bc8a8646cdfa817f0a89c1389
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1a1cb919d48a1c62aa894739651ca55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2017d3bc8a8646cdfa817f0a89c1389
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1a1cb919d48a1c62aa894739651ca55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2017d3bc8a8646cdfa817f0a89c1389
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1a1cb919d48a1c62aa894739651ca55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2017d3bc8a8646cdfa817f0a89c1389
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_00a2bf3b3f9f2c3b122641b32c32af9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d63830ccdc5b459994c27c33545f54aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00a2bf3b3f9f2c3b122641b32c32af9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d63830ccdc5b459994c27c33545f54aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00a2bf3b3f9f2c3b122641b32c32af9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_907eddb316b270a7f94b6f321f2199de(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_860a106f859102541642112a50ca0568(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_907eddb316b270a7f94b6f321f2199de
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_860a106f859102541642112a50ca0568(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_907eddb316b270a7f94b6f321f2199de
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d4e194f7c12721a71a6cd7360e4309d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_95b75e7776b1070292f6bdcf938eab81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4e194f7c12721a71a6cd7360e4309d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95b75e7776b1070292f6bdcf938eab81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4e194f7c12721a71a6cd7360e4309d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d987a3df581b2ba83c41da9a89a12edb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9f1135191d5b55dce875f9a1ffe15ffe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d987a3df581b2ba83c41da9a89a12edb
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1135191d5b55dce875f9a1ffe15ffe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d987a3df581b2ba83c41da9a89a12edb
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6c66ca726f8fe1e68de2d903f271cadc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ebc474438db5b367e387275d17532791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c66ca726f8fe1e68de2d903f271cadc
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33700039982795715], [0.13599033653736115], [0.16089512407779694], [0.4550243616104126], [0.2522077262401581], [0.41581976413726807], [0.06474526226520538], [0.22361089289188385], [0.08131606876850128]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.31189051270484924], [0.1544954627752304], [0.3554002642631531], [0.06358063966035843], [0.38996535539627075], [0.37870699167251587], [0.02283250354230404], [0.03404628112912178], [0.4961751103401184]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_5aed3934271bb07ddcf79df8793cb850(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c66ca726f8fe1e68de2d903f271cadc
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17808885872364044], [0.1587141752243042], [0.2988268733024597], [0.33936357498168945], [0.37306493520736694], [0.3378402590751648], [0.11509011685848236], [0.29118287563323975], [0.28715723752975464]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.08929294347763062], [0.13941359519958496], [0.3334523141384125], [0.2755478620529175], [0.273370623588562], [0.07012606412172318], [0.0644788146018982], [0.1514199823141098], [0.44735172390937805]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_41a1cfeefc894836287b17b5adf9680f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c66ca726f8fe1e68de2d903f271cadc
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40283963084220886], [0.45022642612457275], [0.4985415041446686], [0.32824042439460754], [0.11407893896102905], [0.04877207800745964], [0.24639740586280823], [0.4660728871822357], [0.06258141249418259]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.10231184214353561], [0.21500952541828156], [0.033458199352025986], [0.3819327652454376], [0.053689487278461456], [0.17333781719207764], [0.3545892834663391], [0.06132432818412781], [0.08616641163825989]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_e60462ec6a39f761277e001d6dc05adf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c66ca726f8fe1e68de2d903f271cadc
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07969757914543152], [0.14082278311252594], [0.477491170167923], [0.15222904086112976], [0.2556220293045044], [0.2010723203420639], [0.2752452790737152], [0.257482647895813], [0.3754969537258148]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4645979106426239], [0.151620551943779], [0.23870912194252014], [0.4139777421951294], [0.40845590829849243], [0.21291503310203552], [0.35330283641815186], [0.264354944229126], [0.2503066658973694]], dtype='float32').reshape([9, 1]),
            ]


    
    class PrimitiveOp_4157b191cccf184e2cd14d9b2f49f708(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5517, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[5517, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8288e65e815534c84224095f4a6a86a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4157b191cccf184e2cd14d9b2f49f708
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8288e65e815534c84224095f4a6a86a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4157b191cccf184e2cd14d9b2f49f708
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8288e65e815534c84224095f4a6a86a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4157b191cccf184e2cd14d9b2f49f708
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8288e65e815534c84224095f4a6a86a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4157b191cccf184e2cd14d9b2f49f708
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6], dtype='float32'),
                paddle.static.InputSpec(shape=[6], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_21d39d19f1ccfb75ec5166f09c56fa8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.32048559188842773, 0.04365863651037216, 0.14801615476608276, 0.059532683342695236, 0.1973104625940323, 0.10881417244672775], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_631d0cd45c943eaf10b48dc879df56ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.43007832765579224, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3242815434932709, 0.21491502225399017, 0.4360129237174988, 0.2119390070438385, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_5d07bb00240af7163b208a0d1dde260c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4498527944087982, 0.4890660345554352, 0.20852655172348022, 0.4761844575405121, 0.404381662607193, 0.10783115774393082], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_83a0b0d95d76fce46f1f7e24752b8cc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.43007832765579224, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1643696129322052, 0.4882410168647766, 0.43661245703697205, 0.06498825550079346, 0.49191561341285706, 0.33490511775016785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e4954886a7078b0f82f42ecc89e9bf2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.45263877511024475, 0.18129011988639832, 0.45884010195732117, 0.49139586091041565, 0.3470446765422821, 0.2975061535835266], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_91661b1071efa18a7be8dc167cd98e24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.39286479353904724, 0.4360129237174988, 0.43007832765579224, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3685716390609741, 0.420699805021286, 0.35412517189979553, 0.2729951739311218, 0.31833115220069885, 0.4968334138393402], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_c07af18ccda11f9606ca6c974864c137(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1794, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1794, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd838cbbb557b8f8f35ae978bcf1d22d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c07af18ccda11f9606ca6c974864c137
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd838cbbb557b8f8f35ae978bcf1d22d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c07af18ccda11f9606ca6c974864c137
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd838cbbb557b8f8f35ae978bcf1d22d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c07af18ccda11f9606ca6c974864c137
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd838cbbb557b8f8f35ae978bcf1d22d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c07af18ccda11f9606ca6c974864c137
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_16b24c65895038cef9ae045935825502(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 38, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_49f880f56c95d13808d9162e88426f9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16b24c65895038cef9ae045935825502
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1.000000013351432e-10], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_88bb50f5a129999fbe0431b50f0c16da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2725aac8670ddbd93636903174f8262f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88bb50f5a129999fbe0431b50f0c16da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2725aac8670ddbd93636903174f8262f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_860a106f859102541642112a50ca0568(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_907eddb316b270a7f94b6f321f2199de
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_860a106f859102541642112a50ca0568(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_907eddb316b270a7f94b6f321f2199de
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6a233952cfd0e353ff6fbd32a40fa1c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1504, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1504, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_19090618b2bea383f58d9075184703bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a233952cfd0e353ff6fbd32a40fa1c4
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19090618b2bea383f58d9075184703bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a233952cfd0e353ff6fbd32a40fa1c4
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19090618b2bea383f58d9075184703bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a233952cfd0e353ff6fbd32a40fa1c4
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19090618b2bea383f58d9075184703bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a233952cfd0e353ff6fbd32a40fa1c4
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47e450c0ceab4c338e7791427d80cc03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9601160761ea3127d8587b22969d6fea
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47e450c0ceab4c338e7791427d80cc03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9601160761ea3127d8587b22969d6fea
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4600577d72a8aa2f1d6d4ca4da4f7532(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb3320a5f060ad2659624ce4621d9127
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4600577d72a8aa2f1d6d4ca4da4f7532(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb3320a5f060ad2659624ce4621d9127
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0dc9d2dbf8f3145b9028cfb82363c3f3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b45bd72aa30b276b4da0bbaae060589d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dc9d2dbf8f3145b9028cfb82363c3f3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.24695035815238953]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.1818336695432663]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_1f93dd6fdc3654af784174b278d59cf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dc9d2dbf8f3145b9028cfb82363c3f3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.27610716223716736]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.028087755665183067]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_8cfdb4d9b99b4942ead51e747654496a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dc9d2dbf8f3145b9028cfb82363c3f3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20727156102657318]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.44695910811424255]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_a5bf0129e88b30770cdc236b7d16253a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dc9d2dbf8f3145b9028cfb82363c3f3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.35763055086135864]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3456944525241852]], dtype='float32').reshape([1, 1]),
            ]


    
    class PrimitiveOp_d6c087bf05e769b8eca6b2c5e3a8ea26(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_da5e2df0505b8fd4bef3f1a810ea9c45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6c087bf05e769b8eca6b2c5e3a8ea26
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06345830112695694], [0.37558823823928833], [0.1739027202129364], [0.027879230678081512], [0.49161165952682495], [0.11910757422447205]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17218370735645294], [0.19266913831233978], [0.3696064352989197], [0.3602122664451599], [0.343851238489151], [0.4164700210094452]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_042ef90b813c702e1dfeba279d37b220(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6c087bf05e769b8eca6b2c5e3a8ea26
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42717310786247253], [0.2651657462120056], [0.46184709668159485], [0.13699816167354584], [0.30248284339904785], [0.26277780532836914]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.01432943344116211], [0.07210607081651688], [0.11303866654634476], [0.4184499979019165], [0.4230078458786011], [0.2603096663951874]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_8a5855681c83621ed1150dbaf70984c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6c087bf05e769b8eca6b2c5e3a8ea26
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1136389821767807], [0.3263870179653168], [0.4104270935058594], [0.2049286961555481], [0.05052289366722107], [0.18513862788677216]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06397107243537903], [0.32089540362358093], [0.02683880925178528], [0.38781505823135376], [0.12123280763626099], [0.1445770114660263]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_5e2ac05908e53b035626414e9e68faa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6c087bf05e769b8eca6b2c5e3a8ea26
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3679124414920807], [0.3236524164676666], [0.1917470246553421], [0.0763665959239006], [0.3653966784477234], [0.06940227746963501]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.07160773128271103], [0.0910111665725708], [0.3088380694389343], [0.12214173376560211], [0.39863425493240356], [0.18909907341003418]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_d63830ccdc5b459994c27c33545f54aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00a2bf3b3f9f2c3b122641b32c32af9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d63830ccdc5b459994c27c33545f54aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00a2bf3b3f9f2c3b122641b32c32af9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43c2ad80706215cf0f77acaf4ff89dcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2033d6afb68542ddf01e093b04f571b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43c2ad80706215cf0f77acaf4ff89dcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2033d6afb68542ddf01e093b04f571b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_677ff564133880012a731c116a3ebcf0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2039, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2039, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d8204bd0631a5542f5edeb67b4aef9e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_677ff564133880012a731c116a3ebcf0
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8204bd0631a5542f5edeb67b4aef9e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_677ff564133880012a731c116a3ebcf0
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8204bd0631a5542f5edeb67b4aef9e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_677ff564133880012a731c116a3ebcf0
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8204bd0631a5542f5edeb67b4aef9e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_677ff564133880012a731c116a3ebcf0
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2ae98422f396f81fd09af73cb207e04d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c422380cdaba990c9c69241c39426a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ae98422f396f81fd09af73cb207e04d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c422380cdaba990c9c69241c39426a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ae98422f396f81fd09af73cb207e04d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_363fbed3821f95cd87108f36c94dde9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4584, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4584, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_56bf01abcdc42c4d7eaf443db41ea7b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_363fbed3821f95cd87108f36c94dde9e
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56bf01abcdc42c4d7eaf443db41ea7b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_363fbed3821f95cd87108f36c94dde9e
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56bf01abcdc42c4d7eaf443db41ea7b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_363fbed3821f95cd87108f36c94dde9e
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56bf01abcdc42c4d7eaf443db41ea7b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_363fbed3821f95cd87108f36c94dde9e
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9fe57b8902da1272574b76c27f5141fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1071, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1071, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41085471974861a51d17374f599f201d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9fe57b8902da1272574b76c27f5141fc
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41085471974861a51d17374f599f201d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9fe57b8902da1272574b76c27f5141fc
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41085471974861a51d17374f599f201d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9fe57b8902da1272574b76c27f5141fc
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41085471974861a51d17374f599f201d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9fe57b8902da1272574b76c27f5141fc
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ad01110cfbf8606f3a39c890c444c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_764eeff18c858cdafc393ac6835aa53f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ad01110cfbf8606f3a39c890c444c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_764eeff18c858cdafc393ac6835aa53f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95b75e7776b1070292f6bdcf938eab81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4e194f7c12721a71a6cd7360e4309d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95b75e7776b1070292f6bdcf938eab81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4e194f7c12721a71a6cd7360e4309d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cf57ddadb2a3b6ad327e8099d2aa2ad9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5045ae656b2caa2cadee9666d3f8f862(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf57ddadb2a3b6ad327e8099d2aa2ad9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29145920276641846], [0.07165557891130447], [0.07205324620008469], [0.22017629444599152], [0.1261482983827591]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3746805787086487], [0.2069297879934311], [0.08722388744354248], [0.3281881809234619], [0.10755358636379242]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_bf18dda996e33efd2ba03353c1b8fa6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf57ddadb2a3b6ad327e8099d2aa2ad9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.446043998003006], [0.12132035940885544], [0.0887727364897728], [0.17385192215442657], [0.07668683677911758]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2575170397758484], [0.09787295013666153], [0.30022138357162476], [0.40387892723083496], [0.24005642533302307]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_27fcd4081bc6ce354d565fd3442262a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf57ddadb2a3b6ad327e8099d2aa2ad9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1386314034461975], [0.3368445634841919], [0.4964533746242523], [0.08265111595392227], [0.13507212698459625]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.104369156062603], [0.09522414207458496], [0.19618113338947296], [0.1922953873872757], [0.1039789542555809]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_95bf6c7869b623dceb4012693314149c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf57ddadb2a3b6ad327e8099d2aa2ad9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05733131244778633], [0.1570308655500412], [0.3689388930797577], [0.17223645746707916], [0.2121700644493103]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.4198094606399536], [0.43931514024734497], [0.03426346927881241], [0.41905465722084045], [0.24269208312034607]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_6c85083743241021347f8171e2fbe23a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3578a466474c5a817b5a6b3631fb846
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c85083743241021347f8171e2fbe23a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3578a466474c5a817b5a6b3631fb846
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1135191d5b55dce875f9a1ffe15ffe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d987a3df581b2ba83c41da9a89a12edb
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1135191d5b55dce875f9a1ffe15ffe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d987a3df581b2ba83c41da9a89a12edb
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_945012d46a3c33039c7ca802b362e3be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_42f290d145603f88878f7b3657dd2693(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_945012d46a3c33039c7ca802b362e3be
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42f290d145603f88878f7b3657dd2693(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_945012d46a3c33039c7ca802b362e3be
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ca228298f8cf242687400c9bc0333c81(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2370, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2370, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_31555dc6e154026ad164c4c9c2eab0c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca228298f8cf242687400c9bc0333c81
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31555dc6e154026ad164c4c9c2eab0c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca228298f8cf242687400c9bc0333c81
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31555dc6e154026ad164c4c9c2eab0c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca228298f8cf242687400c9bc0333c81
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31555dc6e154026ad164c4c9c2eab0c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca228298f8cf242687400c9bc0333c81
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_91a6cdd27ea13a1006999f2c1ae1ceb5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2993, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2993, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5232f2576ea2f1a85541125735bbe05e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91a6cdd27ea13a1006999f2c1ae1ceb5
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5232f2576ea2f1a85541125735bbe05e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91a6cdd27ea13a1006999f2c1ae1ceb5
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5232f2576ea2f1a85541125735bbe05e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91a6cdd27ea13a1006999f2c1ae1ceb5
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5232f2576ea2f1a85541125735bbe05e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91a6cdd27ea13a1006999f2c1ae1ceb5
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_73d60a69c9c7491fd854c977a6c360db(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3832, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3832, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b07eef380c7acca0b37c62755edb1c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73d60a69c9c7491fd854c977a6c360db
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b07eef380c7acca0b37c62755edb1c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73d60a69c9c7491fd854c977a6c360db
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b07eef380c7acca0b37c62755edb1c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73d60a69c9c7491fd854c977a6c360db
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b07eef380c7acca0b37c62755edb1c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73d60a69c9c7491fd854c977a6c360db
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c422380cdaba990c9c69241c39426a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ae98422f396f81fd09af73cb207e04d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c422380cdaba990c9c69241c39426a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ae98422f396f81fd09af73cb207e04d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6761ebf88a760162203ae52877640656(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_81e6eea260d21fdb4b04c9403e95802a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6761ebf88a760162203ae52877640656
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41876405477523804], [0.0375184640288353], [0.09816955029964447], [0.42315077781677246]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2992156445980072], [0.462017297744751], [0.11372362822294235], [0.295624315738678]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_3551917feae86b517df816910c4b9806(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6761ebf88a760162203ae52877640656
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13186973333358765], [0.09202171117067337], [0.08312810212373734], [0.2566831707954407]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2604764997959137], [0.1259506195783615], [0.3157579004764557], [0.2131945937871933]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_a704a3ec8d1a55d2801e2207ddb8b44e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6761ebf88a760162203ae52877640656
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10784570127725601], [0.43085747957229614], [0.18588005006313324], [0.29377901554107666]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.4131563901901245], [0.2982136011123657], [0.22558467090129852], [0.03406929969787598]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_ccb4964f12f0f11fc8762bfef5814cc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6761ebf88a760162203ae52877640656
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3623265326023102], [0.025396505370736122], [0.3224317133426666], [0.4281119704246521]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.33433759212493896], [0.12095697224140167], [0.2845987379550934], [0.09898775070905685]], dtype='float32').reshape([4, 1]),
            ]


    
    class PrimitiveOp_ba0fae5c8090eb073bb05c08a8ff80de(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c989959a97c9c07d45f89022e09e6114(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba0fae5c8090eb073bb05c08a8ff80de
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c989959a97c9c07d45f89022e09e6114(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba0fae5c8090eb073bb05c08a8ff80de
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c989959a97c9c07d45f89022e09e6114(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba0fae5c8090eb073bb05c08a8ff80de
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c989959a97c9c07d45f89022e09e6114(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba0fae5c8090eb073bb05c08a8ff80de
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42f290d145603f88878f7b3657dd2693(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_945012d46a3c33039c7ca802b362e3be
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42f290d145603f88878f7b3657dd2693(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_945012d46a3c33039c7ca802b362e3be
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db7b3fa80cbf7b98dd5936a4f1a859e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dbf42f4978cca69a91ba2f31b774922
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db7b3fa80cbf7b98dd5936a4f1a859e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dbf42f4978cca69a91ba2f31b774922
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9028e3aa5d70099eafc6bc0e7ab7b6cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d9fa587a1c258ac2adf41372d0eeae4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9028e3aa5d70099eafc6bc0e7ab7b6cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d9fa587a1c258ac2adf41372d0eeae4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9028e3aa5d70099eafc6bc0e7ab7b6cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_322ac61e4f6836bd5b50cf4b029d4f3a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4181, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4181, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ae38c1d976fa77c53a8a2d99f9d59de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_322ac61e4f6836bd5b50cf4b029d4f3a
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ae38c1d976fa77c53a8a2d99f9d59de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_322ac61e4f6836bd5b50cf4b029d4f3a
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ae38c1d976fa77c53a8a2d99f9d59de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_322ac61e4f6836bd5b50cf4b029d4f3a
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ae38c1d976fa77c53a8a2d99f9d59de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_322ac61e4f6836bd5b50cf4b029d4f3a
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d9fa587a1c258ac2adf41372d0eeae4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9028e3aa5d70099eafc6bc0e7ab7b6cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d9fa587a1c258ac2adf41372d0eeae4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9028e3aa5d70099eafc6bc0e7ab7b6cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86c4c3277305dd6ddff1af0f5290982f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86c4c3277305dd6ddff1af0f5290982f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3dad9b4bf6a3008eb6f309b0341e9227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3dad9b4bf6a3008eb6f309b0341e9227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8468c8c23b232dd0cb87269ce1f2b387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8468c8c23b232dd0cb87269ce1f2b387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d67af0770c1a44070f76e873e81e675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d67af0770c1a44070f76e873e81e675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a86f8642b02d880fcb8a4a380e88ab00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a86f8642b02d880fcb8a4a380e88ab00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d90e35866eba82b9904def88da3cb1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d90e35866eba82b9904def88da3cb1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86c4c3277305dd6ddff1af0f5290982f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86c4c3277305dd6ddff1af0f5290982f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9630ce511e3c8b9c83e9bf9784841f8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9630ce511e3c8b9c83e9bf9784841f8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0dd44a76c7a4e8447bf424768eadb13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0dd44a76c7a4e8447bf424768eadb13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60d72502e4412260834c96c050cd3cde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60d72502e4412260834c96c050cd3cde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60d72502e4412260834c96c050cd3cde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60d72502e4412260834c96c050cd3cde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc1d3bdade8d147f6188485410ffdb17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc1d3bdade8d147f6188485410ffdb17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f27bc3cf81be12de4e3e0f186b99f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f27bc3cf81be12de4e3e0f186b99f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e27a01469fb2ab3e752bb1143a535d1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e27a01469fb2ab3e752bb1143a535d1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9e6a88155731b4eda272425ecf74d32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9e6a88155731b4eda272425ecf74d32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2afbed2592e5cc4a5551e79e22801598(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33700039982795715], [0.13599033653736115], [0.16089512407779694], [0.4550243616104126], [0.2522077262401581], [0.41581976413726807], [0.06474526226520538], [0.22361089289188385], [0.08131606876850128]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.31189051270484924], [0.1544954627752304], [0.3554002642631531], [0.06358063966035843], [0.38996535539627075], [0.37870699167251587], [0.02283250354230404], [0.03404628112912178], [0.4961751103401184]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_60024d8db5bc3243b383d9aa54b39b5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17808885872364044], [0.1587141752243042], [0.2988268733024597], [0.33936357498168945], [0.37306493520736694], [0.3378402590751648], [0.11509011685848236], [0.29118287563323975], [0.28715723752975464]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.08929294347763062], [0.13941359519958496], [0.3334523141384125], [0.2755478620529175], [0.273370623588562], [0.07012606412172318], [0.0644788146018982], [0.1514199823141098], [0.44735172390937805]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_d035f9ebefd35d55b2b35a1b09073b3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40283963084220886], [0.45022642612457275], [0.4985415041446686], [0.32824042439460754], [0.11407893896102905], [0.04877207800745964], [0.24639740586280823], [0.4660728871822357], [0.06258141249418259]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.10231184214353561], [0.21500952541828156], [0.033458199352025986], [0.3819327652454376], [0.053689487278461456], [0.17333781719207764], [0.3545892834663391], [0.06132432818412781], [0.08616641163825989]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_dbafd909fd12b0df2209061f14c98568(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07969757914543152], [0.14082278311252594], [0.477491170167923], [0.15222904086112976], [0.2556220293045044], [0.2010723203420639], [0.2752452790737152], [0.257482647895813], [0.3754969537258148]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4645979106426239], [0.151620551943779], [0.23870912194252014], [0.4139777421951294], [0.40845590829849243], [0.21291503310203552], [0.35330283641815186], [0.264354944229126], [0.2503066658973694]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_f749eda516ee41042422216c21ec0750(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f749eda516ee41042422216c21ec0750(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f749eda516ee41042422216c21ec0750(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f749eda516ee41042422216c21ec0750(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2c7e92d0a23f8980620768017275982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.32048559188842773, 0.04365863651037216, 0.14801615476608276, 0.059532683342695236, 0.1973104625940323, 0.10881417244672775], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c194de1968e31cedec708a7595e5077d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.43007832765579224, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3242815434932709, 0.21491502225399017, 0.4360129237174988, 0.2119390070438385, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_a1613e2d37c0b30331bd3db47a459e46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4498527944087982, 0.4890660345554352, 0.20852655172348022, 0.4761844575405121, 0.404381662607193, 0.10783115774393082], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_16cf1af4cbe311e51bdf562f63b55af4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.43007832765579224, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1643696129322052, 0.4882410168647766, 0.43661245703697205, 0.06498825550079346, 0.49191561341285706, 0.33490511775016785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_fed2be89f18f829301a603fe2323a009(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.45263877511024475, 0.18129011988639832, 0.45884010195732117, 0.49139586091041565, 0.3470446765422821, 0.2975061535835266], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8b52a3427a5a405c47e167e6f3f1e18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.39286479353904724, 0.4360129237174988, 0.43007832765579224, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3685716390609741, 0.420699805021286, 0.35412517189979553, 0.2729951739311218, 0.31833115220069885, 0.4968334138393402], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_064ee668f1c201098d0de78b0e4ff4d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_064ee668f1c201098d0de78b0e4ff4d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_064ee668f1c201098d0de78b0e4ff4d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_064ee668f1c201098d0de78b0e4ff4d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_23fab028d3c0b148ab9db02fafb72657(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d7baa494ab40bffa0fed68b25b6be217(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23fab028d3c0b148ab9db02fafb72657
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1.000000013351432e-10], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3d67af0770c1a44070f76e873e81e675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d67af0770c1a44070f76e873e81e675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f27bc3cf81be12de4e3e0f186b99f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f27bc3cf81be12de4e3e0f186b99f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_302d8e9c3b1cddbfbeb706c3d431a58f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_302d8e9c3b1cddbfbeb706c3d431a58f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_302d8e9c3b1cddbfbeb706c3d431a58f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_302d8e9c3b1cddbfbeb706c3d431a58f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d90e35866eba82b9904def88da3cb1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d90e35866eba82b9904def88da3cb1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3dad9b4bf6a3008eb6f309b0341e9227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3dad9b4bf6a3008eb6f309b0341e9227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f4aeb903741b82161ab397e2c158a34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.24695035815238953]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.1818336695432663]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_1c536259d63dbe2cc5f67ff2c18847df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.27610716223716736]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.028087755665183067]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_85643ea3b519a973808a08713d188cfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20727156102657318]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.44695910811424255]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ca6c191223432795bb978e2d9746b9be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.35763055086135864]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3456944525241852]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_43bb972d96de9dd14ce449c0af2a7105(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06345830112695694], [0.37558823823928833], [0.1739027202129364], [0.027879230678081512], [0.49161165952682495], [0.11910757422447205]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17218370735645294], [0.19266913831233978], [0.3696064352989197], [0.3602122664451599], [0.343851238489151], [0.4164700210094452]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_9781246a6316269b6468672fc48f0842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42717310786247253], [0.2651657462120056], [0.46184709668159485], [0.13699816167354584], [0.30248284339904785], [0.26277780532836914]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.01432943344116211], [0.07210607081651688], [0.11303866654634476], [0.4184499979019165], [0.4230078458786011], [0.2603096663951874]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_2e6a4aad420411bbd3accbd1870de0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1136389821767807], [0.3263870179653168], [0.4104270935058594], [0.2049286961555481], [0.05052289366722107], [0.18513862788677216]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06397107243537903], [0.32089540362358093], [0.02683880925178528], [0.38781505823135376], [0.12123280763626099], [0.1445770114660263]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_4dd7d3b9a622587d2bfdc56bf3907fdc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3679124414920807], [0.3236524164676666], [0.1917470246553421], [0.0763665959239006], [0.3653966784477234], [0.06940227746963501]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.07160773128271103], [0.0910111665725708], [0.3088380694389343], [0.12214173376560211], [0.39863425493240356], [0.18909907341003418]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_bc1d3bdade8d147f6188485410ffdb17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc1d3bdade8d147f6188485410ffdb17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8468c8c23b232dd0cb87269ce1f2b387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8468c8c23b232dd0cb87269ce1f2b387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d177e0d80e4ec97173c7d262c06b5b20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d177e0d80e4ec97173c7d262c06b5b20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d177e0d80e4ec97173c7d262c06b5b20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d177e0d80e4ec97173c7d262c06b5b20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e95c45b5ce0c8bce6a4c17a7df65c4d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e95c45b5ce0c8bce6a4c17a7df65c4d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed5497f1d722ec765c5173d816df4c76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed5497f1d722ec765c5173d816df4c76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed5497f1d722ec765c5173d816df4c76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed5497f1d722ec765c5173d816df4c76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3673ca5c5da781fdd12c860951b444ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3673ca5c5da781fdd12c860951b444ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3673ca5c5da781fdd12c860951b444ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3673ca5c5da781fdd12c860951b444ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a86f8642b02d880fcb8a4a380e88ab00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a86f8642b02d880fcb8a4a380e88ab00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e27a01469fb2ab3e752bb1143a535d1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e27a01469fb2ab3e752bb1143a535d1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fae8c5c800da71f1f6c0feab207b4a4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29145920276641846], [0.07165557891130447], [0.07205324620008469], [0.22017629444599152], [0.1261482983827591]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3746805787086487], [0.2069297879934311], [0.08722388744354248], [0.3281881809234619], [0.10755358636379242]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_ff3cbdc45a27bb5ce2addc2657624b9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.446043998003006], [0.12132035940885544], [0.0887727364897728], [0.17385192215442657], [0.07668683677911758]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2575170397758484], [0.09787295013666153], [0.30022138357162476], [0.40387892723083496], [0.24005642533302307]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_629f68d6d3b3baef4d5c068ae60e5a31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1386314034461975], [0.3368445634841919], [0.4964533746242523], [0.08265111595392227], [0.13507212698459625]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.104369156062603], [0.09522414207458496], [0.19618113338947296], [0.1922953873872757], [0.1039789542555809]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_33e42703e8b2c861a5ed1b60c9e53a17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05733131244778633], [0.1570308655500412], [0.3689388930797577], [0.17223645746707916], [0.2121700644493103]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.4198094606399536], [0.43931514024734497], [0.03426346927881241], [0.41905465722084045], [0.24269208312034607]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_c0dd44a76c7a4e8447bf424768eadb13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0dd44a76c7a4e8447bf424768eadb13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9e6a88155731b4eda272425ecf74d32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9e6a88155731b4eda272425ecf74d32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db140cfb159ed29f63fe09694cb8a42d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db140cfb159ed29f63fe09694cb8a42d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_389004671194ebfe9ead23708b0c3f91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_389004671194ebfe9ead23708b0c3f91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_389004671194ebfe9ead23708b0c3f91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_389004671194ebfe9ead23708b0c3f91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea7257d6cac28638645675bfa3465b1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea7257d6cac28638645675bfa3465b1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea7257d6cac28638645675bfa3465b1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea7257d6cac28638645675bfa3465b1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fcb40825adb28c2da8ba7a4f840dc7d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fcb40825adb28c2da8ba7a4f840dc7d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fcb40825adb28c2da8ba7a4f840dc7d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fcb40825adb28c2da8ba7a4f840dc7d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e95c45b5ce0c8bce6a4c17a7df65c4d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e95c45b5ce0c8bce6a4c17a7df65c4d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57b08af9e9fe107bfb775ad293c1900b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41876405477523804], [0.0375184640288353], [0.09816955029964447], [0.42315077781677246]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2992156445980072], [0.462017297744751], [0.11372362822294235], [0.295624315738678]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_2b29dc3d7f378c27f9fca38e3d3500d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13186973333358765], [0.09202171117067337], [0.08312810212373734], [0.2566831707954407]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2604764997959137], [0.1259506195783615], [0.3157579004764557], [0.2131945937871933]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_9cf9b730f21ce06af1a5a099f6559338(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10784570127725601], [0.43085747957229614], [0.18588005006313324], [0.29377901554107666]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.4131563901901245], [0.2982136011123657], [0.22558467090129852], [0.03406929969787598]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_a229717e6b9be30be85c21065e9701fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3623265326023102], [0.025396505370736122], [0.3224317133426666], [0.4281119704246521]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.33433759212493896], [0.12095697224140167], [0.2845987379550934], [0.09898775070905685]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_1a547b31b69a84b8c56d167e52ecac6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a547b31b69a84b8c56d167e52ecac6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a547b31b69a84b8c56d167e52ecac6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a547b31b69a84b8c56d167e52ecac6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db140cfb159ed29f63fe09694cb8a42d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db140cfb159ed29f63fe09694cb8a42d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9630ce511e3c8b9c83e9bf9784841f8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9630ce511e3c8b9c83e9bf9784841f8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a1cc7361323b317054a26dd43f98f92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a1cc7361323b317054a26dd43f98f92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30dad79b1907e493faabea16a1c28ce9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30dad79b1907e493faabea16a1c28ce9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30dad79b1907e493faabea16a1c28ce9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30dad79b1907e493faabea16a1c28ce9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a1cc7361323b317054a26dd43f98f92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a1cc7361323b317054a26dd43f98f92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_753e012d67b402ee09ab5cd4cb492dcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()