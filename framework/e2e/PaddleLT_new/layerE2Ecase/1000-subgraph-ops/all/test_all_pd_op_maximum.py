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


    class TestPrimitiveOp_7a7b8c05f71d2cbdae163cb9bd79353c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a7b8c05f71d2cbdae163cb9bd79353c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a7b8c05f71d2cbdae163cb9bd79353c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a7b8c05f71d2cbdae163cb9bd79353c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_0d895622fbf6fa217dc99b1a4c342238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18618053197860718], [0.007991598919034004], [0.08734893053770065], [0.48177430033683777], [0.4414879083633423], [0.1868826150894165], [0.3090613782405853], [0.07356120645999908], [0.4638229012489319]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0328356958925724], [0.042280279099941254], [0.08102629333734512], [0.3775736689567566], [0.0860084742307663], [0.3348342180252075], [0.45164355635643005], [0.08683238923549652], [0.4006982743740082]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_1c490606fb573f161db85d4d99b9458f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25806623697280884], [0.013965434394776821], [0.2514757812023163], [0.4430234134197235], [0.21071745455265045], [0.002510953461751342], [0.4495491087436676], [0.22793996334075928], [0.3716922998428345]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.3526066541671753], [0.4370092451572418], [0.11843360215425491], [0.4964841306209564], [0.09898043423891068], [0.1466875672340393], [0.21377909183502197], [0.2298479974269867], [0.3915681838989258]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_59c4ec02fe39722a1e48f7bcca310d7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36666956543922424], [0.36908185482025146], [0.2961925268173218], [0.035441603511571884], [0.4349591135978699], [0.2363598495721817], [0.18706205487251282], [0.06219051405787468], [0.03899889066815376]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4386983811855316], [0.046371277421712875], [0.0077659436501562595], [0.29862847924232483], [0.4404768645763397], [0.39816227555274963], [0.3323768377304077], [0.047508757561445236], [0.04453969746828079]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_c90ff18e44d3a514def7761b71d37e34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4806118905544281], [0.338043212890625], [0.493656724691391], [0.3843192160129547], [0.4271245300769806], [0.173434779047966], [0.3057922422885895], [0.23234502971172333], [0.08419803529977798]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.42142167687416077], [0.39432990550994873], [0.012163503095507622], [0.2716471254825592], [0.08528527617454529], [0.09083577990531921], [0.41013792157173157], [0.3947860300540924], [0.2945002317428589]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_495400b453b3cc1523f70c0b8e79cdbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_495400b453b3cc1523f70c0b8e79cdbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_495400b453b3cc1523f70c0b8e79cdbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_495400b453b3cc1523f70c0b8e79cdbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_2d411673c4882d906ea47251e7188ba5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1420275717973709, 0.13624942302703857, 0.2611750066280365, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.32582366466522217, 0.04505692422389984, 0.4173981845378876, 0.0506727397441864, 0.004974461626261473, 0.0021464754827320576], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_152f39e8efc63802492ab830c1792f60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.27147242426872253, 0.34147074818611145, 0.1349279284477234, 0.04442029446363449, 0.05202075466513634, 0.2485688030719757], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c5548647ec2fd75d568cc620530119d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1420275717973709, 0.13624942302703857, 0.2611750066280365, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.04602406919002533, 0.3307744562625885, 0.4149071276187897, 0.13126017153263092, 0.009036889299750328, 0.3169717490673065], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_2d41a780115546261562782ed7d8e105(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.373994380235672, 0.10453741252422333, 0.016055766493082047, 0.4658578038215637, 0.22361966967582703, 0.03576983883976936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_d5c8b1da3113b93eb35d71af09905f40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32582366466522217, 0.13624942302703857, 0.4173981845378876, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.35163626074790955, 0.4952782392501831, 0.19085991382598877, 0.13665537536144257, 0.2634378671646118, 0.18630926311016083], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e10a78b80c02ab2048e1f8e016e854e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.23692236840724945, 0.3351680040359497, 0.10397005081176758, 0.09553638845682144, 0.20359452068805695, 0.42669540643692017], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c5e0f263ad962fa33d118fd3241be394(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5e0f263ad962fa33d118fd3241be394(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5e0f263ad962fa33d118fd3241be394(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5e0f263ad962fa33d118fd3241be394(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_7e5611ae6dca08dfe3e26d290e00f5e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e5611ae6dca08dfe3e26d290e00f5e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e5611ae6dca08dfe3e26d290e00f5e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e5611ae6dca08dfe3e26d290e00f5e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_f358b3e820af823712a77376b44a0351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4937298595905304]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.01672634482383728]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_b4effa065d3c13757f6c470b09072b1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08633378893136978]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.21023254096508026]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ab7728d8ae09fca420018791320eaae3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4828311502933502]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.2893354296684265]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_c3958c61c2f1e0bb0fbbdc355f29fa93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4335895776748657]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.23339727520942688]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_b5aad81d26e2a75a56bd9cc023b0d39d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13643445074558258], [0.08216515928506851], [0.41876524686813354], [0.46803197264671326], [0.20365557074546814], [0.2550865709781647]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0025779781863093376], [0.41118323802948], [0.2937028706073761], [0.40923434495925903], [0.060637570917606354], [0.34649741649627686]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_9485836781f19025d98715710ce024ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06172452121973038], [0.29318130016326904], [0.42264029383659363], [0.03228634223341942], [0.35276615619659424], [0.1730971336364746]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.35616353154182434], [0.3317866325378418], [0.39316296577453613], [0.4681013226509094], [0.0868077278137207], [0.19587846100330353]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_58a58c564eadc5fcbd33a33c143c444d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13170580565929413], [0.058484263718128204], [0.06756104528903961], [0.14288948476314545], [0.4329535663127899], [0.38953283429145813]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.20599016547203064], [0.29832562804222107], [0.4217703342437744], [0.34355056285858154], [0.2057957798242569], [0.31516075134277344]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_08676cd75335d1ed458e6e627ddfbbad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.48313620686531067], [0.30361756682395935], [0.09585190564393997], [0.27821820974349976], [0.4517127275466919], [0.20359814167022705]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.34790247678756714], [0.01755298487842083], [0.2810487151145935], [0.3574141561985016], [0.4414989948272705], [0.43758612871170044]], dtype='float32').reshape([6, 1]),
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


    class TestPrimitiveOp_bc20ffccfc67a360e65ac22cecebbe9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc20ffccfc67a360e65ac22cecebbe9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc20ffccfc67a360e65ac22cecebbe9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc20ffccfc67a360e65ac22cecebbe9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_3f6385bef8395d31f3c4267ccc638164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f6385bef8395d31f3c4267ccc638164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f6385bef8395d31f3c4267ccc638164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f6385bef8395d31f3c4267ccc638164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec7ad6d625a00f6213fa5fec4caaa6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec7ad6d625a00f6213fa5fec4caaa6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec7ad6d625a00f6213fa5fec4caaa6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec7ad6d625a00f6213fa5fec4caaa6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_804a6f6ee6d60167059e7cb57a7f3ba6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3187001049518585], [0.395175576210022], [0.4700484275817871], [0.33523043990135193], [0.4927985966205597]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1435127556324005], [0.09152258932590485], [0.2088557481765747], [0.3625732958316803], [0.48419061303138733]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_331b40624926940cd31cb995e0398282(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40533360838890076], [0.32245928049087524], [0.2967213988304138], [0.3857642710208893], [0.0886489748954773]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.49802151322364807], [0.3020210564136505], [0.07553115487098694], [0.10842543095350266], [0.21820658445358276]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_fa28dac4b0ba68fa8b1c2c2d466a464a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29793399572372437], [0.45982950925827026], [0.041210729628801346], [0.08048146218061447], [0.11273333430290222]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.44977208971977234], [0.4644966721534729], [0.2582728862762451], [0.2239593118429184], [0.40969327092170715]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_87a96b30a786988b76bce62d4bb3fa5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29028844833374023], [0.16028515994548798], [0.474911093711853], [0.49212008714675903], [0.08536812663078308]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.26832517981529236], [0.4581025242805481], [0.2240569293498993], [0.3378629684448242], [0.006186048965901136]], dtype='float32').reshape([5, 1]),
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


    class TestPrimitiveOp_d1b5a6e72b27a36c786468777493483d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1b5a6e72b27a36c786468777493483d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1b5a6e72b27a36c786468777493483d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1b5a6e72b27a36c786468777493483d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f5e0d18f14bec53403424cf8e884d02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f5e0d18f14bec53403424cf8e884d02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f5e0d18f14bec53403424cf8e884d02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f5e0d18f14bec53403424cf8e884d02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68736802a8d5106f4c57a387b0ffe841(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68736802a8d5106f4c57a387b0ffe841(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68736802a8d5106f4c57a387b0ffe841(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68736802a8d5106f4c57a387b0ffe841(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_045eb3fbe9c801840fd1ff3359425a85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41484373807907104], [0.09475763142108917], [0.012042115442454815], [0.26613038778305054]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.3469667136669159], [0.09736814349889755], [0.3636853098869324], [0.14231333136558533]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_d86303dcf0b19487826ba2a65dfafacf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4367034435272217], [0.03958968445658684], [0.29002854228019714], [0.24420864880084991]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.441641241312027], [0.2988758683204651], [0.3646744191646576], [0.2739860415458679]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_3ea53dee22eac041bb2efdc5466511b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2870025336742401], [0.021595124155282974], [0.4148816764354706], [0.3877279460430145]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.09640353918075562], [0.25064805150032043], [0.09111341089010239], [0.12773194909095764]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_d1051c135ca7ae97375283124d755680(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10462348163127899], [0.3512554168701172], [0.08861131221055984], [0.4113050699234009]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.22977420687675476], [0.061657778918743134], [0.2658151686191559], [0.3008590340614319]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_24f3198de014f3789047b31329087eaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24f3198de014f3789047b31329087eaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24f3198de014f3789047b31329087eaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24f3198de014f3789047b31329087eaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_43736cae67767e8b64167dd66be0d84d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43736cae67767e8b64167dd66be0d84d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43736cae67767e8b64167dd66be0d84d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43736cae67767e8b64167dd66be0d84d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_6bb57eb5f41bb488c9f617856e72e9c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dcdb3232bf83be7fc8569b95c18730c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bb57eb5f41bb488c9f617856e72e9c8
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcdb3232bf83be7fc8569b95c18730c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bb57eb5f41bb488c9f617856e72e9c8
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcdb3232bf83be7fc8569b95c18730c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bb57eb5f41bb488c9f617856e72e9c8
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcdb3232bf83be7fc8569b95c18730c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bb57eb5f41bb488c9f617856e72e9c8
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_a50179ec0c0931794545e6a782263889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c66ca726f8fe1e68de2d903f271cadc
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18618053197860718], [0.007991598919034004], [0.08734893053770065], [0.48177430033683777], [0.4414879083633423], [0.1868826150894165], [0.3090613782405853], [0.07356120645999908], [0.4638229012489319]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0328356958925724], [0.042280279099941254], [0.08102629333734512], [0.3775736689567566], [0.0860084742307663], [0.3348342180252075], [0.45164355635643005], [0.08683238923549652], [0.4006982743740082]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_9359e4ec92155473798b28978be6763e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c66ca726f8fe1e68de2d903f271cadc
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25806623697280884], [0.013965434394776821], [0.2514757812023163], [0.4430234134197235], [0.21071745455265045], [0.002510953461751342], [0.4495491087436676], [0.22793996334075928], [0.3716922998428345]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.3526066541671753], [0.4370092451572418], [0.11843360215425491], [0.4964841306209564], [0.09898043423891068], [0.1466875672340393], [0.21377909183502197], [0.2298479974269867], [0.3915681838989258]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_e1f0f10b73fe9371d1bb34a455375e00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c66ca726f8fe1e68de2d903f271cadc
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36666956543922424], [0.36908185482025146], [0.2961925268173218], [0.035441603511571884], [0.4349591135978699], [0.2363598495721817], [0.18706205487251282], [0.06219051405787468], [0.03899889066815376]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4386983811855316], [0.046371277421712875], [0.0077659436501562595], [0.29862847924232483], [0.4404768645763397], [0.39816227555274963], [0.3323768377304077], [0.047508757561445236], [0.04453969746828079]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_a164eca886520c49b0a0d752a6b76cbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c66ca726f8fe1e68de2d903f271cadc
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4806118905544281], [0.338043212890625], [0.493656724691391], [0.3843192160129547], [0.4271245300769806], [0.173434779047966], [0.3057922422885895], [0.23234502971172333], [0.08419803529977798]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.42142167687416077], [0.39432990550994873], [0.012163503095507622], [0.2716471254825592], [0.08528527617454529], [0.09083577990531921], [0.41013792157173157], [0.3947860300540924], [0.2945002317428589]], dtype='float32').reshape([9, 1]),
            ]


    
    class PrimitiveOp_fdb6e085e8d04568506df4a4681903ca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5454, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[5454, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_689e7fd9aeffae23f065b838e733ab33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdb6e085e8d04568506df4a4681903ca
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_689e7fd9aeffae23f065b838e733ab33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdb6e085e8d04568506df4a4681903ca
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_689e7fd9aeffae23f065b838e733ab33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdb6e085e8d04568506df4a4681903ca
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_689e7fd9aeffae23f065b838e733ab33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdb6e085e8d04568506df4a4681903ca
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_1e5244c9cc3e40c453ad938a689ea9fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1420275717973709, 0.13624942302703857, 0.2611750066280365, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.32582366466522217, 0.04505692422389984, 0.4173981845378876, 0.0506727397441864, 0.004974461626261473, 0.0021464754827320576], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_07f6198ef1b4ab35d16eb7a5ddb7a66f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.27147242426872253, 0.34147074818611145, 0.1349279284477234, 0.04442029446363449, 0.05202075466513634, 0.2485688030719757], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_433769462618f72ad2be426d61d57ea5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1420275717973709, 0.13624942302703857, 0.2611750066280365, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.04602406919002533, 0.3307744562625885, 0.4149071276187897, 0.13126017153263092, 0.009036889299750328, 0.3169717490673065], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_29861eda3bb595788e1f23ca2ba84e69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.373994380235672, 0.10453741252422333, 0.016055766493082047, 0.4658578038215637, 0.22361966967582703, 0.03576983883976936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_91064e81f7c3cdbbd561ee21f974c491(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32582366466522217, 0.13624942302703857, 0.4173981845378876, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.35163626074790955, 0.4952782392501831, 0.19085991382598877, 0.13665537536144257, 0.2634378671646118, 0.18630926311016083], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_71a06d9d4af05555e4ffb64205aee333(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.23692236840724945, 0.3351680040359497, 0.10397005081176758, 0.09553638845682144, 0.20359452068805695, 0.42669540643692017], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_8c48da4c88b5ddd8c32639f95c65d842(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1722, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1722, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f33f211cc015f20bf8fbd717baf46711(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c48da4c88b5ddd8c32639f95c65d842
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f33f211cc015f20bf8fbd717baf46711(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c48da4c88b5ddd8c32639f95c65d842
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f33f211cc015f20bf8fbd717baf46711(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c48da4c88b5ddd8c32639f95c65d842
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f33f211cc015f20bf8fbd717baf46711(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c48da4c88b5ddd8c32639f95c65d842
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_fe629dec6e2f9323941c869b2d5d2c5d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1518, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1518, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3713293efda26db4e17dae0f90ba8ac1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe629dec6e2f9323941c869b2d5d2c5d
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3713293efda26db4e17dae0f90ba8ac1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe629dec6e2f9323941c869b2d5d2c5d
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3713293efda26db4e17dae0f90ba8ac1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe629dec6e2f9323941c869b2d5d2c5d
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3713293efda26db4e17dae0f90ba8ac1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe629dec6e2f9323941c869b2d5d2c5d
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_db30cd368a2bf29130503df9b4fea1ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dc9d2dbf8f3145b9028cfb82363c3f3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4937298595905304]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.01672634482383728]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_78329c73d8eb4f78eafbf0e3246ded02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dc9d2dbf8f3145b9028cfb82363c3f3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08633378893136978]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.21023254096508026]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_89762d61cc83613b96d380a0013671d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dc9d2dbf8f3145b9028cfb82363c3f3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4828311502933502]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.2893354296684265]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_8c8fd9e5d42d9359dc5a43db2578f883(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dc9d2dbf8f3145b9028cfb82363c3f3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4335895776748657]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.23339727520942688]], dtype='float32').reshape([1, 1]),
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


    class TestPrimitiveOp_a38aacda5a9cd124db22905bc1cf4e4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6c087bf05e769b8eca6b2c5e3a8ea26
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13643445074558258], [0.08216515928506851], [0.41876524686813354], [0.46803197264671326], [0.20365557074546814], [0.2550865709781647]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0025779781863093376], [0.41118323802948], [0.2937028706073761], [0.40923434495925903], [0.060637570917606354], [0.34649741649627686]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_cc6437848e0446b195c7b2e9d4d76b5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6c087bf05e769b8eca6b2c5e3a8ea26
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06172452121973038], [0.29318130016326904], [0.42264029383659363], [0.03228634223341942], [0.35276615619659424], [0.1730971336364746]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.35616353154182434], [0.3317866325378418], [0.39316296577453613], [0.4681013226509094], [0.0868077278137207], [0.19587846100330353]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_e118cbc5b22712fe1bbebd8b13afd4f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6c087bf05e769b8eca6b2c5e3a8ea26
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13170580565929413], [0.058484263718128204], [0.06756104528903961], [0.14288948476314545], [0.4329535663127899], [0.38953283429145813]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.20599016547203064], [0.29832562804222107], [0.4217703342437744], [0.34355056285858154], [0.2057957798242569], [0.31516075134277344]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_1316c7919fb71d0ead56af8a7a68a2b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6c087bf05e769b8eca6b2c5e3a8ea26
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.48313620686531067], [0.30361756682395935], [0.09585190564393997], [0.27821820974349976], [0.4517127275466919], [0.20359814167022705]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.34790247678756714], [0.01755298487842083], [0.2810487151145935], [0.3574141561985016], [0.4414989948272705], [0.43758612871170044]], dtype='float32').reshape([6, 1]),
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


    
    class PrimitiveOp_d5c898e8d9fc1cead23685be85d48d76(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2133, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2133, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_49966f177a9da7ebb0dce9f5f4989a3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5c898e8d9fc1cead23685be85d48d76
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49966f177a9da7ebb0dce9f5f4989a3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5c898e8d9fc1cead23685be85d48d76
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49966f177a9da7ebb0dce9f5f4989a3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5c898e8d9fc1cead23685be85d48d76
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49966f177a9da7ebb0dce9f5f4989a3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5c898e8d9fc1cead23685be85d48d76
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_ec37e1274a08c8fa0ac2f0f185d9d15c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4631, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4631, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_efe0f5411cda9117a435c2eb213784e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec37e1274a08c8fa0ac2f0f185d9d15c
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efe0f5411cda9117a435c2eb213784e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec37e1274a08c8fa0ac2f0f185d9d15c
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efe0f5411cda9117a435c2eb213784e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec37e1274a08c8fa0ac2f0f185d9d15c
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efe0f5411cda9117a435c2eb213784e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec37e1274a08c8fa0ac2f0f185d9d15c
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9fd6efa8610b6f20abf2fde9bfdde1d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1039, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1039, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a832971597a6d84466d335771286e854(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9fd6efa8610b6f20abf2fde9bfdde1d6
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a832971597a6d84466d335771286e854(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9fd6efa8610b6f20abf2fde9bfdde1d6
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a832971597a6d84466d335771286e854(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9fd6efa8610b6f20abf2fde9bfdde1d6
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a832971597a6d84466d335771286e854(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9fd6efa8610b6f20abf2fde9bfdde1d6
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_a41410f337ddd5ee489d416c2b1d64dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf57ddadb2a3b6ad327e8099d2aa2ad9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3187001049518585], [0.395175576210022], [0.4700484275817871], [0.33523043990135193], [0.4927985966205597]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1435127556324005], [0.09152258932590485], [0.2088557481765747], [0.3625732958316803], [0.48419061303138733]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_874647a7a442b95a7aa445a8c2019727(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf57ddadb2a3b6ad327e8099d2aa2ad9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40533360838890076], [0.32245928049087524], [0.2967213988304138], [0.3857642710208893], [0.0886489748954773]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.49802151322364807], [0.3020210564136505], [0.07553115487098694], [0.10842543095350266], [0.21820658445358276]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_646b40d557232f3535795186b616a5d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf57ddadb2a3b6ad327e8099d2aa2ad9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29793399572372437], [0.45982950925827026], [0.041210729628801346], [0.08048146218061447], [0.11273333430290222]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.44977208971977234], [0.4644966721534729], [0.2582728862762451], [0.2239593118429184], [0.40969327092170715]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_c77cfae05190488446faca0fc9dca9a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf57ddadb2a3b6ad327e8099d2aa2ad9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29028844833374023], [0.16028515994548798], [0.474911093711853], [0.49212008714675903], [0.08536812663078308]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.26832517981529236], [0.4581025242805481], [0.2240569293498993], [0.3378629684448242], [0.006186048965901136]], dtype='float32').reshape([5, 1]),
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


    
    class PrimitiveOp_c7143d344c891d51339c0de9698cf9f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2318, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2318, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a492b9db2e35a005f254a74d40ba4b43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7143d344c891d51339c0de9698cf9f5
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a492b9db2e35a005f254a74d40ba4b43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7143d344c891d51339c0de9698cf9f5
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a492b9db2e35a005f254a74d40ba4b43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7143d344c891d51339c0de9698cf9f5
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a492b9db2e35a005f254a74d40ba4b43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7143d344c891d51339c0de9698cf9f5
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_48943002db64242254219defbc761262(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2961, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2961, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_03b888b817d49eb45d35ca76c6278904(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48943002db64242254219defbc761262
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03b888b817d49eb45d35ca76c6278904(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48943002db64242254219defbc761262
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03b888b817d49eb45d35ca76c6278904(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48943002db64242254219defbc761262
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03b888b817d49eb45d35ca76c6278904(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48943002db64242254219defbc761262
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_611996c73df4bcad28775211c8a7e734(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3739, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3739, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d441dcd69c0a002e29927f6eef753e99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_611996c73df4bcad28775211c8a7e734
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d441dcd69c0a002e29927f6eef753e99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_611996c73df4bcad28775211c8a7e734
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d441dcd69c0a002e29927f6eef753e99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_611996c73df4bcad28775211c8a7e734
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d441dcd69c0a002e29927f6eef753e99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_611996c73df4bcad28775211c8a7e734
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_b83fecf4d87adc37565b88bc7b864ac3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6761ebf88a760162203ae52877640656
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41484373807907104], [0.09475763142108917], [0.012042115442454815], [0.26613038778305054]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.3469667136669159], [0.09736814349889755], [0.3636853098869324], [0.14231333136558533]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_723d2b6e1b6b74890e78f2c6df39407a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6761ebf88a760162203ae52877640656
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4367034435272217], [0.03958968445658684], [0.29002854228019714], [0.24420864880084991]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.441641241312027], [0.2988758683204651], [0.3646744191646576], [0.2739860415458679]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_60b8e04e296a6d49af84457c65d01735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6761ebf88a760162203ae52877640656
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2870025336742401], [0.021595124155282974], [0.4148816764354706], [0.3877279460430145]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.09640353918075562], [0.25064805150032043], [0.09111341089010239], [0.12773194909095764]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f9f9e4f420a3eb532b4a3bedfa78a30a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6761ebf88a760162203ae52877640656
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10462348163127899], [0.3512554168701172], [0.08861131221055984], [0.4113050699234009]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.22977420687675476], [0.061657778918743134], [0.2658151686191559], [0.3008590340614319]], dtype='float32').reshape([4, 1]),
            ]


    
    class PrimitiveOp_a805e17ec5e83e17756de1a7db47ae2a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2013, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2013, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c84ef39799437d41e7509c2ca13801df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a805e17ec5e83e17756de1a7db47ae2a
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c84ef39799437d41e7509c2ca13801df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a805e17ec5e83e17756de1a7db47ae2a
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c84ef39799437d41e7509c2ca13801df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a805e17ec5e83e17756de1a7db47ae2a
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c84ef39799437d41e7509c2ca13801df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a805e17ec5e83e17756de1a7db47ae2a
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_20653cca423c35847d946f582f79bd4e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4177, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4177, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9742914eee143a3b9ba340a052f26c4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20653cca423c35847d946f582f79bd4e
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9742914eee143a3b9ba340a052f26c4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20653cca423c35847d946f582f79bd4e
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9742914eee143a3b9ba340a052f26c4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20653cca423c35847d946f582f79bd4e
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9742914eee143a3b9ba340a052f26c4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20653cca423c35847d946f582f79bd4e
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_2b330ba45829e6ac9943baea7af6c942(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330ba45829e6ac9943baea7af6c942(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330ba45829e6ac9943baea7af6c942(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330ba45829e6ac9943baea7af6c942(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_0d895622fbf6fa217dc99b1a4c342238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18618053197860718], [0.007991598919034004], [0.08734893053770065], [0.48177430033683777], [0.4414879083633423], [0.1868826150894165], [0.3090613782405853], [0.07356120645999908], [0.4638229012489319]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0328356958925724], [0.042280279099941254], [0.08102629333734512], [0.3775736689567566], [0.0860084742307663], [0.3348342180252075], [0.45164355635643005], [0.08683238923549652], [0.4006982743740082]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_1c490606fb573f161db85d4d99b9458f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25806623697280884], [0.013965434394776821], [0.2514757812023163], [0.4430234134197235], [0.21071745455265045], [0.002510953461751342], [0.4495491087436676], [0.22793996334075928], [0.3716922998428345]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.3526066541671753], [0.4370092451572418], [0.11843360215425491], [0.4964841306209564], [0.09898043423891068], [0.1466875672340393], [0.21377909183502197], [0.2298479974269867], [0.3915681838989258]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_59c4ec02fe39722a1e48f7bcca310d7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36666956543922424], [0.36908185482025146], [0.2961925268173218], [0.035441603511571884], [0.4349591135978699], [0.2363598495721817], [0.18706205487251282], [0.06219051405787468], [0.03899889066815376]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4386983811855316], [0.046371277421712875], [0.0077659436501562595], [0.29862847924232483], [0.4404768645763397], [0.39816227555274963], [0.3323768377304077], [0.047508757561445236], [0.04453969746828079]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_c90ff18e44d3a514def7761b71d37e34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4806118905544281], [0.338043212890625], [0.493656724691391], [0.3843192160129547], [0.4271245300769806], [0.173434779047966], [0.3057922422885895], [0.23234502971172333], [0.08419803529977798]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.42142167687416077], [0.39432990550994873], [0.012163503095507622], [0.2716471254825592], [0.08528527617454529], [0.09083577990531921], [0.41013792157173157], [0.3947860300540924], [0.2945002317428589]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_e80fe7a2c6702726e459993a8fb4fa24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e80fe7a2c6702726e459993a8fb4fa24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e80fe7a2c6702726e459993a8fb4fa24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e80fe7a2c6702726e459993a8fb4fa24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d411673c4882d906ea47251e7188ba5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1420275717973709, 0.13624942302703857, 0.2611750066280365, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.32582366466522217, 0.04505692422389984, 0.4173981845378876, 0.0506727397441864, 0.004974461626261473, 0.0021464754827320576], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_152f39e8efc63802492ab830c1792f60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.27147242426872253, 0.34147074818611145, 0.1349279284477234, 0.04442029446363449, 0.05202075466513634, 0.2485688030719757], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c5548647ec2fd75d568cc620530119d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1420275717973709, 0.13624942302703857, 0.2611750066280365, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.04602406919002533, 0.3307744562625885, 0.4149071276187897, 0.13126017153263092, 0.009036889299750328, 0.3169717490673065], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_2d41a780115546261562782ed7d8e105(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.373994380235672, 0.10453741252422333, 0.016055766493082047, 0.4658578038215637, 0.22361966967582703, 0.03576983883976936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_d5c8b1da3113b93eb35d71af09905f40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32582366466522217, 0.13624942302703857, 0.4173981845378876, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.35163626074790955, 0.4952782392501831, 0.19085991382598877, 0.13665537536144257, 0.2634378671646118, 0.18630926311016083], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e10a78b80c02ab2048e1f8e016e854e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.23692236840724945, 0.3351680040359497, 0.10397005081176758, 0.09553638845682144, 0.20359452068805695, 0.42669540643692017], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e10ce1d283abd38f882ce08d3a815ba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e10ce1d283abd38f882ce08d3a815ba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e10ce1d283abd38f882ce08d3a815ba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e10ce1d283abd38f882ce08d3a815ba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_49c1a2b02375574479adf89928ae2bb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49c1a2b02375574479adf89928ae2bb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49c1a2b02375574479adf89928ae2bb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49c1a2b02375574479adf89928ae2bb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_f358b3e820af823712a77376b44a0351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4937298595905304]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.01672634482383728]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_b4effa065d3c13757f6c470b09072b1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08633378893136978]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.21023254096508026]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ab7728d8ae09fca420018791320eaae3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4828311502933502]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.2893354296684265]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_c3958c61c2f1e0bb0fbbdc355f29fa93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4335895776748657]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.23339727520942688]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_b5aad81d26e2a75a56bd9cc023b0d39d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13643445074558258], [0.08216515928506851], [0.41876524686813354], [0.46803197264671326], [0.20365557074546814], [0.2550865709781647]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0025779781863093376], [0.41118323802948], [0.2937028706073761], [0.40923434495925903], [0.060637570917606354], [0.34649741649627686]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_9485836781f19025d98715710ce024ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06172452121973038], [0.29318130016326904], [0.42264029383659363], [0.03228634223341942], [0.35276615619659424], [0.1730971336364746]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.35616353154182434], [0.3317866325378418], [0.39316296577453613], [0.4681013226509094], [0.0868077278137207], [0.19587846100330353]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_58a58c564eadc5fcbd33a33c143c444d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13170580565929413], [0.058484263718128204], [0.06756104528903961], [0.14288948476314545], [0.4329535663127899], [0.38953283429145813]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.20599016547203064], [0.29832562804222107], [0.4217703342437744], [0.34355056285858154], [0.2057957798242569], [0.31516075134277344]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_08676cd75335d1ed458e6e627ddfbbad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.48313620686531067], [0.30361756682395935], [0.09585190564393997], [0.27821820974349976], [0.4517127275466919], [0.20359814167022705]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.34790247678756714], [0.01755298487842083], [0.2810487151145935], [0.3574141561985016], [0.4414989948272705], [0.43758612871170044]], dtype='float32').reshape([6, 1]),
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


    class TestPrimitiveOp_dd74e159ce8248cab722ec6fc061f073(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd74e159ce8248cab722ec6fc061f073(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd74e159ce8248cab722ec6fc061f073(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd74e159ce8248cab722ec6fc061f073(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_c6f1e697678d3019a90c683eabcfe278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6f1e697678d3019a90c683eabcfe278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6f1e697678d3019a90c683eabcfe278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6f1e697678d3019a90c683eabcfe278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c7f1f5eb4b14c88d206145b428f6a44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c7f1f5eb4b14c88d206145b428f6a44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c7f1f5eb4b14c88d206145b428f6a44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c7f1f5eb4b14c88d206145b428f6a44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_804a6f6ee6d60167059e7cb57a7f3ba6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3187001049518585], [0.395175576210022], [0.4700484275817871], [0.33523043990135193], [0.4927985966205597]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1435127556324005], [0.09152258932590485], [0.2088557481765747], [0.3625732958316803], [0.48419061303138733]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_331b40624926940cd31cb995e0398282(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40533360838890076], [0.32245928049087524], [0.2967213988304138], [0.3857642710208893], [0.0886489748954773]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.49802151322364807], [0.3020210564136505], [0.07553115487098694], [0.10842543095350266], [0.21820658445358276]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_fa28dac4b0ba68fa8b1c2c2d466a464a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29793399572372437], [0.45982950925827026], [0.041210729628801346], [0.08048146218061447], [0.11273333430290222]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.44977208971977234], [0.4644966721534729], [0.2582728862762451], [0.2239593118429184], [0.40969327092170715]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_87a96b30a786988b76bce62d4bb3fa5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29028844833374023], [0.16028515994548798], [0.474911093711853], [0.49212008714675903], [0.08536812663078308]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.26832517981529236], [0.4581025242805481], [0.2240569293498993], [0.3378629684448242], [0.006186048965901136]], dtype='float32').reshape([5, 1]),
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


    class TestPrimitiveOp_775109cad403252ed6a1f73769983f54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_775109cad403252ed6a1f73769983f54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_775109cad403252ed6a1f73769983f54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_775109cad403252ed6a1f73769983f54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f61a26f484041ff92a19648f76bdfd65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f61a26f484041ff92a19648f76bdfd65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f61a26f484041ff92a19648f76bdfd65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f61a26f484041ff92a19648f76bdfd65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbd2ce10d3d8403905432f0c4ca050e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbd2ce10d3d8403905432f0c4ca050e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbd2ce10d3d8403905432f0c4ca050e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbd2ce10d3d8403905432f0c4ca050e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_045eb3fbe9c801840fd1ff3359425a85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41484373807907104], [0.09475763142108917], [0.012042115442454815], [0.26613038778305054]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.3469667136669159], [0.09736814349889755], [0.3636853098869324], [0.14231333136558533]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_d86303dcf0b19487826ba2a65dfafacf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4367034435272217], [0.03958968445658684], [0.29002854228019714], [0.24420864880084991]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.441641241312027], [0.2988758683204651], [0.3646744191646576], [0.2739860415458679]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_3ea53dee22eac041bb2efdc5466511b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2870025336742401], [0.021595124155282974], [0.4148816764354706], [0.3877279460430145]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.09640353918075562], [0.25064805150032043], [0.09111341089010239], [0.12773194909095764]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_d1051c135ca7ae97375283124d755680(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10462348163127899], [0.3512554168701172], [0.08861131221055984], [0.4113050699234009]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.22977420687675476], [0.061657778918743134], [0.2658151686191559], [0.3008590340614319]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_57e46fa1cdee371b16a4551b0b6f6634(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57e46fa1cdee371b16a4551b0b6f6634(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57e46fa1cdee371b16a4551b0b6f6634(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57e46fa1cdee371b16a4551b0b6f6634(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_6fc40cdc1f3fa23b7012e1ae4c063903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fc40cdc1f3fa23b7012e1ae4c063903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fc40cdc1f3fa23b7012e1ae4c063903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fc40cdc1f3fa23b7012e1ae4c063903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
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