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


    class TestPrimitiveOp_9df1f0b9564d42cbeb6e8268686c5295(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9df1f0b9564d42cbeb6e8268686c5295(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9df1f0b9564d42cbeb6e8268686c5295(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9df1f0b9564d42cbeb6e8268686c5295(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_8aa0c7ef1632de247d4d7314fb16ad57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2521946430206299], [0.19013427197933197], [0.40820974111557007], [0.3767457604408264], [0.20149092376232147], [0.07500499486923218], [0.328750878572464], [0.31747785210609436], [0.24313530325889587]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4038843512535095], [0.44629916548728943], [0.008235457353293896], [0.14164923131465912], [0.18166889250278473], [0.179908886551857], [0.3791463375091553], [0.4934486150741577], [0.07347828894853592]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_5ed230dabd1be8b4889c6412889b4649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04966363683342934], [0.19880470633506775], [0.04978129267692566], [0.042689915746450424], [0.1670447289943695], [0.3255453109741211], [0.22168521583080292], [0.11604061722755432], [0.43528881669044495]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.22095853090286255], [0.2874982953071594], [0.0034736385568976402], [0.2015986442565918], [0.023722784593701363], [0.057638633996248245], [0.05427054315805435], [0.3369775414466858], [0.09985696524381638]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_9a160e5db521b1e3605d3c2e01550335(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41298598051071167], [0.48954424262046814], [0.2974533438682556], [0.04537849500775337], [0.4212323725223541], [0.14318227767944336], [0.2683485746383667], [0.40157246589660645], [0.34450793266296387]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.39238229393959045], [0.158734992146492], [0.4127102792263031], [0.41945791244506836], [0.29879409074783325], [0.013535123318433762], [0.1018265038728714], [0.21279259026050568], [0.030626868829131126]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_95f959d2068734b61c9187f7cf182653(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36636999249458313], [0.2677699625492096], [0.015402782708406448], [0.36943402886390686], [0.43435823917388916], [0.2168995440006256], [0.1261843889951706], [0.05905040726065636], [0.30280032753944397]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.09990614652633667], [0.32696646451950073], [0.2620046138763428], [0.3387252688407898], [0.2716210186481476], [0.1099492609500885], [0.29109078645706177], [0.030494073405861855], [0.40721428394317627]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_0a6495616af013bb1d217867d17115b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a6495616af013bb1d217867d17115b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a6495616af013bb1d217867d17115b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a6495616af013bb1d217867d17115b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_9e9403775293db79ff3edd4f9816e864(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.040052223950624466, 0.026079408824443817, 0.223669171333313, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.33742555975914, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.09696357697248459], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8186bf40a33b503427fdc547afd28b4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1651548445224762, 0.4702686071395874, 0.36698785424232483, 0.0644494965672493, 0.4465831220149994, 0.07492171972990036], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1810450553894043, 0.2536962032318115, 0.2332039326429367, 0.22083275020122528, 0.2845509946346283, 0.10143007338047028], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_427c442e2c15f63fe31f9bebba9451da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.040052223950624466, 0.026079408824443817, 0.223669171333313, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3705940544605255, 0.04870932921767235, 0.07290997356176376, 0.22891433537006378, 0.06531837582588196, 0.44855570793151855], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_941843276467250c9de028bb6a29cf42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1651548445224762, 0.4702686071395874, 0.36698785424232483, 0.0644494965672493, 0.4465831220149994, 0.07492171972990036], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3314341604709625, 0.10902168601751328, 0.04308721795678139, 0.134864941239357, 0.4397190809249878, 0.21771910786628723], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_d584aa188dcdf5ec1ce51f9e50749342(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2057289332151413, 0.09391630440950394, 0.047374628484249115, 0.2254459410905838, 0.18349812924861908, 0.12224911153316498], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e763d8ebd5508c3fdfdf6acc5bed375a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1810450553894043, 0.4702686071395874, 0.36698785424232483, 0.22083275020122528, 0.4465831220149994, 0.10143007338047028], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2081550508737564, 0.41452664136886597, 0.23729932308197021, 0.11266523599624634, 0.0390239953994751, 0.285847932100296], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_2e570e7e38da843491ab29ea582afb65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e570e7e38da843491ab29ea582afb65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e570e7e38da843491ab29ea582afb65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e570e7e38da843491ab29ea582afb65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_6577df85151cc6cc2d7ff8c93fd030d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6577df85151cc6cc2d7ff8c93fd030d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6577df85151cc6cc2d7ff8c93fd030d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6577df85151cc6cc2d7ff8c93fd030d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_5b8346acc90367e80a0919b98bcca0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2060057669878006]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.05126293748617172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_8a88f0105bc59254df47595d57c28f13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.26949211955070496]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3662005662918091]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_939b55f9f031baf0dad674069a88d657(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20535515248775482]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.08478929847478867]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_2ab0dd8f08b35e3896c052d4d33c2393(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.491769939661026]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.40578794479370117]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_7ba2af42cb15128d464f7bf0285f568a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3966229259967804], [0.2040838748216629], [0.2837878465652466], [0.1662617325782776], [0.08668506145477295], [0.3346942663192749]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17277514934539795], [0.4306377172470093], [0.21807396411895752], [0.11639563739299774], [0.04053834080696106], [0.47490552067756653]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_eb25e27ea97540770848167957a1f358(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49894431233406067], [0.3844832479953766], [0.21334876120090485], [0.25372934341430664], [0.006601519882678986], [0.32043617963790894]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.09358027577400208], [0.3427625000476837], [0.1698751002550125], [0.2064223736524582], [0.3752198815345764], [0.4434046745300293]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_d921db52310cd88555896973d9eb8438(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05556122213602066], [0.3532122075557709], [0.09650922566652298], [0.2863064706325531], [0.4242008626461029], [0.07322079688310623]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4361984133720398], [0.14439257979393005], [0.49514085054397583], [0.16213607788085938], [0.43578216433525085], [0.15172600746154785]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_37503a87ea2dbb6b57cbf62464919f0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04935036972165108], [0.401021808385849], [0.15970346331596375], [0.36091819405555725], [0.45432034134864807], [0.24458400905132294]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4443901479244232], [0.49830520153045654], [0.14623917639255524], [0.0019387512002140284], [0.4178217053413391], [0.1282878816127777]], dtype='float32').reshape([6, 1]),
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


    class TestPrimitiveOp_c3348862dc179da6d696f900299a4cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3348862dc179da6d696f900299a4cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3348862dc179da6d696f900299a4cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3348862dc179da6d696f900299a4cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_e07144acc97b186722caa3278853f75f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e07144acc97b186722caa3278853f75f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e07144acc97b186722caa3278853f75f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e07144acc97b186722caa3278853f75f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7f9a8f3d8746de9d44d8f7a08eb9ad0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7f9a8f3d8746de9d44d8f7a08eb9ad0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7f9a8f3d8746de9d44d8f7a08eb9ad0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7f9a8f3d8746de9d44d8f7a08eb9ad0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_95062009c83656883b46bd9e07b362b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08599822223186493], [0.21465438604354858], [0.023872841149568558], [0.1336366832256317], [0.12001485377550125]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.19767306745052338], [0.23302686214447021], [0.31559082865715027], [0.44974565505981445], [0.12936343252658844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_249fc5b8d2459701053bb60bec2908ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.31906890869140625], [0.02482573315501213], [0.439250111579895], [0.08620838075876236], [0.13948924839496613]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.22683300077915192], [0.04625708982348442], [0.1838442087173462], [0.45197582244873047], [0.287341833114624]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_9f2f95c778ee0c2679955ec3967407c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16905468702316284], [0.07408490031957626], [0.13317765295505524], [0.36288416385650635], [0.0633009746670723]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.4030136466026306], [0.48801735043525696], [0.401704341173172], [0.3282714784145355], [0.1866285353899002]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_03f13b7678906969890f4f4113ff3456(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15759168565273285], [0.09588633477687836], [0.37041568756103516], [0.4901740849018097], [0.2975691258907318]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2669719457626343], [0.3428363800048828], [0.49236804246902466], [0.149748757481575], [0.15492382645606995]], dtype='float32').reshape([5, 1]),
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


    class TestPrimitiveOp_f95282295a730af58ec98869121aa43e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f95282295a730af58ec98869121aa43e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f95282295a730af58ec98869121aa43e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f95282295a730af58ec98869121aa43e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_960357b287e958377662215faf1459e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_960357b287e958377662215faf1459e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_960357b287e958377662215faf1459e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_960357b287e958377662215faf1459e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa3bdc525f182b058a751b0415e4a71b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa3bdc525f182b058a751b0415e4a71b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa3bdc525f182b058a751b0415e4a71b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa3bdc525f182b058a751b0415e4a71b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_e5c9af3eeee6c0cc1a65f3be64a54dc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33240923285484314], [0.013569344766438007], [0.04725205898284912], [0.23816898465156555]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.40112707018852234], [0.16776657104492188], [0.05913810804486275], [0.45849156379699707]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_7f6f16d38d9cc9827132f25b073c3cc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.32093942165374756], [0.10365208238363266], [0.052328769117593765], [0.030598077923059464]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.21341432631015778], [0.16157600283622742], [0.17247040569782257], [0.42010706663131714]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_726e0ad432dcbdfe08000a7365713ca4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49404314160346985], [0.2501605749130249], [0.0006863751332275569], [0.08632545918226242]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.14522317051887512], [0.43382689356803894], [0.1448354572057724], [0.3873145282268524]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f9683f882f192217d7404013d0df090b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16857455670833588], [0.48723578453063965], [0.0007394644781015813], [0.12745331227779388]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.28212907910346985], [0.440621554851532], [0.13090020418167114], [0.48451167345046997]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_851ae5ad90dd3b946046488b6cc53975(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_851ae5ad90dd3b946046488b6cc53975(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_851ae5ad90dd3b946046488b6cc53975(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_851ae5ad90dd3b946046488b6cc53975(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_749b2b9bcb8de0cfecb184bbd425e277(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_749b2b9bcb8de0cfecb184bbd425e277(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_749b2b9bcb8de0cfecb184bbd425e277(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_749b2b9bcb8de0cfecb184bbd425e277(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc15e4f5f2b2c4edb811759a96c01d7c
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_e8347fca1efa714cea7ea494c2596d7d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1723, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1723, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1343941cb392871e81f749f264541d37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8347fca1efa714cea7ea494c2596d7d
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1343941cb392871e81f749f264541d37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8347fca1efa714cea7ea494c2596d7d
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1343941cb392871e81f749f264541d37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8347fca1efa714cea7ea494c2596d7d
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1343941cb392871e81f749f264541d37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8347fca1efa714cea7ea494c2596d7d
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_419c14cd140b8ea465f2132cb1a59340(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c66ca726f8fe1e68de2d903f271cadc
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2521946430206299], [0.19013427197933197], [0.40820974111557007], [0.3767457604408264], [0.20149092376232147], [0.07500499486923218], [0.328750878572464], [0.31747785210609436], [0.24313530325889587]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4038843512535095], [0.44629916548728943], [0.008235457353293896], [0.14164923131465912], [0.18166889250278473], [0.179908886551857], [0.3791463375091553], [0.4934486150741577], [0.07347828894853592]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_edd9e7e3198178c797594c0a5a566295(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c66ca726f8fe1e68de2d903f271cadc
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04966363683342934], [0.19880470633506775], [0.04978129267692566], [0.042689915746450424], [0.1670447289943695], [0.3255453109741211], [0.22168521583080292], [0.11604061722755432], [0.43528881669044495]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.22095853090286255], [0.2874982953071594], [0.0034736385568976402], [0.2015986442565918], [0.023722784593701363], [0.057638633996248245], [0.05427054315805435], [0.3369775414466858], [0.09985696524381638]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_40ee2f07b730312af09fb633e52df08b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c66ca726f8fe1e68de2d903f271cadc
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41298598051071167], [0.48954424262046814], [0.2974533438682556], [0.04537849500775337], [0.4212323725223541], [0.14318227767944336], [0.2683485746383667], [0.40157246589660645], [0.34450793266296387]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.39238229393959045], [0.158734992146492], [0.4127102792263031], [0.41945791244506836], [0.29879409074783325], [0.013535123318433762], [0.1018265038728714], [0.21279259026050568], [0.030626868829131126]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_97af8ca40d48bf04087ca14fcfcaa8b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c66ca726f8fe1e68de2d903f271cadc
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36636999249458313], [0.2677699625492096], [0.015402782708406448], [0.36943402886390686], [0.43435823917388916], [0.2168995440006256], [0.1261843889951706], [0.05905040726065636], [0.30280032753944397]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.09990614652633667], [0.32696646451950073], [0.2620046138763428], [0.3387252688407898], [0.2716210186481476], [0.1099492609500885], [0.29109078645706177], [0.030494073405861855], [0.40721428394317627]], dtype='float32').reshape([9, 1]),
            ]


    
    class PrimitiveOp_262ef3702f3f478ff758399c97bfeea9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5498, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[5498, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8cd7e79f52b49b3e823547e4dac31ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_262ef3702f3f478ff758399c97bfeea9
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cd7e79f52b49b3e823547e4dac31ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_262ef3702f3f478ff758399c97bfeea9
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cd7e79f52b49b3e823547e4dac31ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_262ef3702f3f478ff758399c97bfeea9
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cd7e79f52b49b3e823547e4dac31ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_262ef3702f3f478ff758399c97bfeea9
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_91f852d3e3cd777cce776eb5f8dcba3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.040052223950624466, 0.026079408824443817, 0.223669171333313, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.33742555975914, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.09696357697248459], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_38c365f52ec3e4da27e751f1790bd65f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1651548445224762, 0.4702686071395874, 0.36698785424232483, 0.0644494965672493, 0.4465831220149994, 0.07492171972990036], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1810450553894043, 0.2536962032318115, 0.2332039326429367, 0.22083275020122528, 0.2845509946346283, 0.10143007338047028], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_d52588ad4ac705008dd5ad2fb62d5f0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.040052223950624466, 0.026079408824443817, 0.223669171333313, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3705940544605255, 0.04870932921767235, 0.07290997356176376, 0.22891433537006378, 0.06531837582588196, 0.44855570793151855], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_cc7c69c42e2a90bebdc28ca72931965d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1651548445224762, 0.4702686071395874, 0.36698785424232483, 0.0644494965672493, 0.4465831220149994, 0.07492171972990036], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3314341604709625, 0.10902168601751328, 0.04308721795678139, 0.134864941239357, 0.4397190809249878, 0.21771910786628723], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_bf54e91938da4517af3856e7acc3b3ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2057289332151413, 0.09391630440950394, 0.047374628484249115, 0.2254459410905838, 0.18349812924861908, 0.12224911153316498], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c77cb6cffb5bff0e6c71c8d14c968773(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3132d91897b64bdf24a3dc980fb906d4
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1810450553894043, 0.4702686071395874, 0.36698785424232483, 0.22083275020122528, 0.4465831220149994, 0.10143007338047028], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2081550508737564, 0.41452664136886597, 0.23729932308197021, 0.11266523599624634, 0.0390239953994751, 0.285847932100296], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_4be76e0fc5e55df87a6f2251a320d927(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1759, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1759, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9505f26ff7fd73d99591d9bee7d0fc04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4be76e0fc5e55df87a6f2251a320d927
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9505f26ff7fd73d99591d9bee7d0fc04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4be76e0fc5e55df87a6f2251a320d927
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9505f26ff7fd73d99591d9bee7d0fc04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4be76e0fc5e55df87a6f2251a320d927
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9505f26ff7fd73d99591d9bee7d0fc04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4be76e0fc5e55df87a6f2251a320d927
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_d4d52f8c9d8de83967e39a174be57c35(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1538, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1538, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_27145356f1c1f7836095f2e03df0c93b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4d52f8c9d8de83967e39a174be57c35
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27145356f1c1f7836095f2e03df0c93b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4d52f8c9d8de83967e39a174be57c35
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27145356f1c1f7836095f2e03df0c93b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4d52f8c9d8de83967e39a174be57c35
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27145356f1c1f7836095f2e03df0c93b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4d52f8c9d8de83967e39a174be57c35
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_9ab03d638ca5c21659912af405a8297c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dc9d2dbf8f3145b9028cfb82363c3f3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2060057669878006]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.05126293748617172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_4c9de987b17ba5d14d68accd861fcb74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dc9d2dbf8f3145b9028cfb82363c3f3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.26949211955070496]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3662005662918091]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_643ac943033799cbf86747995a306e89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dc9d2dbf8f3145b9028cfb82363c3f3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20535515248775482]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.08478929847478867]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_9f3ff04004f01ca663e0c35a7f36a0fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dc9d2dbf8f3145b9028cfb82363c3f3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.491769939661026]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.40578794479370117]], dtype='float32').reshape([1, 1]),
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


    class TestPrimitiveOp_699d148598e86d9390376ea620893f3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6c087bf05e769b8eca6b2c5e3a8ea26
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3966229259967804], [0.2040838748216629], [0.2837878465652466], [0.1662617325782776], [0.08668506145477295], [0.3346942663192749]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17277514934539795], [0.4306377172470093], [0.21807396411895752], [0.11639563739299774], [0.04053834080696106], [0.47490552067756653]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_7874152d63755bfb30aa66277bf0148e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6c087bf05e769b8eca6b2c5e3a8ea26
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49894431233406067], [0.3844832479953766], [0.21334876120090485], [0.25372934341430664], [0.006601519882678986], [0.32043617963790894]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.09358027577400208], [0.3427625000476837], [0.1698751002550125], [0.2064223736524582], [0.3752198815345764], [0.4434046745300293]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_29add31976163dfa3a82d0733f5e5f7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6c087bf05e769b8eca6b2c5e3a8ea26
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05556122213602066], [0.3532122075557709], [0.09650922566652298], [0.2863064706325531], [0.4242008626461029], [0.07322079688310623]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4361984133720398], [0.14439257979393005], [0.49514085054397583], [0.16213607788085938], [0.43578216433525085], [0.15172600746154785]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_c3c9da921a851f1a664b6472f115eea0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6c087bf05e769b8eca6b2c5e3a8ea26
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04935036972165108], [0.401021808385849], [0.15970346331596375], [0.36091819405555725], [0.45432034134864807], [0.24458400905132294]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4443901479244232], [0.49830520153045654], [0.14623917639255524], [0.0019387512002140284], [0.4178217053413391], [0.1282878816127777]], dtype='float32').reshape([6, 1]),
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


    
    class PrimitiveOp_aa7ec3f9bfac837a6658e22e347d4c03(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2135, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2135, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_08470249fd0a4a420b992fc425725129(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa7ec3f9bfac837a6658e22e347d4c03
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08470249fd0a4a420b992fc425725129(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa7ec3f9bfac837a6658e22e347d4c03
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08470249fd0a4a420b992fc425725129(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa7ec3f9bfac837a6658e22e347d4c03
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08470249fd0a4a420b992fc425725129(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa7ec3f9bfac837a6658e22e347d4c03
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_accb5b05807a36f2211950c37cb60ee7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4590, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4590, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_42020b689383df6f346b6980c9ab9589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_accb5b05807a36f2211950c37cb60ee7
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42020b689383df6f346b6980c9ab9589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_accb5b05807a36f2211950c37cb60ee7
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42020b689383df6f346b6980c9ab9589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_accb5b05807a36f2211950c37cb60ee7
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42020b689383df6f346b6980c9ab9589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_accb5b05807a36f2211950c37cb60ee7
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_da354945d8c2d37882fb07cf883e0142(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1042, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1042, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_53d0065778486b7090f2434b3c7acfec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da354945d8c2d37882fb07cf883e0142
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53d0065778486b7090f2434b3c7acfec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da354945d8c2d37882fb07cf883e0142
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53d0065778486b7090f2434b3c7acfec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da354945d8c2d37882fb07cf883e0142
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53d0065778486b7090f2434b3c7acfec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da354945d8c2d37882fb07cf883e0142
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_438da1f45d3ca457b94887eed7a5dc71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf57ddadb2a3b6ad327e8099d2aa2ad9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08599822223186493], [0.21465438604354858], [0.023872841149568558], [0.1336366832256317], [0.12001485377550125]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.19767306745052338], [0.23302686214447021], [0.31559082865715027], [0.44974565505981445], [0.12936343252658844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_3a0b41e0e0b8ca1d187367f1cc066644(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf57ddadb2a3b6ad327e8099d2aa2ad9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.31906890869140625], [0.02482573315501213], [0.439250111579895], [0.08620838075876236], [0.13948924839496613]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.22683300077915192], [0.04625708982348442], [0.1838442087173462], [0.45197582244873047], [0.287341833114624]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_c03d70aca975d610d3dd86018bd758b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf57ddadb2a3b6ad327e8099d2aa2ad9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16905468702316284], [0.07408490031957626], [0.13317765295505524], [0.36288416385650635], [0.0633009746670723]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.4030136466026306], [0.48801735043525696], [0.401704341173172], [0.3282714784145355], [0.1866285353899002]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_ce818826f45f50fbf7b7ab342b488626(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf57ddadb2a3b6ad327e8099d2aa2ad9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15759168565273285], [0.09588633477687836], [0.37041568756103516], [0.4901740849018097], [0.2975691258907318]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2669719457626343], [0.3428363800048828], [0.49236804246902466], [0.149748757481575], [0.15492382645606995]], dtype='float32').reshape([5, 1]),
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


    
    class PrimitiveOp_865ce85171f074edc901b707e910ce4c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2339, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2339, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f9a867fc60889e964eb25d35cfc884e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_865ce85171f074edc901b707e910ce4c
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9a867fc60889e964eb25d35cfc884e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_865ce85171f074edc901b707e910ce4c
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9a867fc60889e964eb25d35cfc884e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_865ce85171f074edc901b707e910ce4c
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9a867fc60889e964eb25d35cfc884e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_865ce85171f074edc901b707e910ce4c
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2712c2e9fab1f81c6d63101b9176deee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3063, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3063, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_753c8cf80b0546643ce54508c0306b04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2712c2e9fab1f81c6d63101b9176deee
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_753c8cf80b0546643ce54508c0306b04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2712c2e9fab1f81c6d63101b9176deee
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_753c8cf80b0546643ce54508c0306b04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2712c2e9fab1f81c6d63101b9176deee
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_753c8cf80b0546643ce54508c0306b04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2712c2e9fab1f81c6d63101b9176deee
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7777bbb8401c0c77b71f0cb02324dd17(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3822, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3822, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1955766239d0f335794db4141a6947f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7777bbb8401c0c77b71f0cb02324dd17
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1955766239d0f335794db4141a6947f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7777bbb8401c0c77b71f0cb02324dd17
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1955766239d0f335794db4141a6947f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7777bbb8401c0c77b71f0cb02324dd17
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1955766239d0f335794db4141a6947f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7777bbb8401c0c77b71f0cb02324dd17
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_bbe0d70027e953367ee042f1f1ae8d62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6761ebf88a760162203ae52877640656
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33240923285484314], [0.013569344766438007], [0.04725205898284912], [0.23816898465156555]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.40112707018852234], [0.16776657104492188], [0.05913810804486275], [0.45849156379699707]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_5cfef61856af79fea5d1abb87f2c1658(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6761ebf88a760162203ae52877640656
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.32093942165374756], [0.10365208238363266], [0.052328769117593765], [0.030598077923059464]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.21341432631015778], [0.16157600283622742], [0.17247040569782257], [0.42010706663131714]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_dc519b73b4c205f417df20bd51ab0741(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6761ebf88a760162203ae52877640656
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49404314160346985], [0.2501605749130249], [0.0006863751332275569], [0.08632545918226242]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.14522317051887512], [0.43382689356803894], [0.1448354572057724], [0.3873145282268524]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_0b41d622fea94a7dd5d68eae5ddad6e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6761ebf88a760162203ae52877640656
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16857455670833588], [0.48723578453063965], [0.0007394644781015813], [0.12745331227779388]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.28212907910346985], [0.440621554851532], [0.13090020418167114], [0.48451167345046997]], dtype='float32').reshape([4, 1]),
            ]


    
    class PrimitiveOp_868b71e58d11a5fd7ea8045cfd8494ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2057, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2057, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_995af720c7a3634161e4c5ca36246fd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_868b71e58d11a5fd7ea8045cfd8494ce
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_995af720c7a3634161e4c5ca36246fd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_868b71e58d11a5fd7ea8045cfd8494ce
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_995af720c7a3634161e4c5ca36246fd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_868b71e58d11a5fd7ea8045cfd8494ce
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_995af720c7a3634161e4c5ca36246fd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_868b71e58d11a5fd7ea8045cfd8494ce
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_ead554563e3b5ded88e3ebbf12e36342(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4189, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4189, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_76a4835cb5ad3b4da6a04fd0344d019a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ead554563e3b5ded88e3ebbf12e36342
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76a4835cb5ad3b4da6a04fd0344d019a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ead554563e3b5ded88e3ebbf12e36342
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76a4835cb5ad3b4da6a04fd0344d019a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ead554563e3b5ded88e3ebbf12e36342
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76a4835cb5ad3b4da6a04fd0344d019a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ead554563e3b5ded88e3ebbf12e36342
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_289023efd3c9faf3c47b2f2d5ce7a006(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_289023efd3c9faf3c47b2f2d5ce7a006(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_289023efd3c9faf3c47b2f2d5ce7a006(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_289023efd3c9faf3c47b2f2d5ce7a006(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_8aa0c7ef1632de247d4d7314fb16ad57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2521946430206299], [0.19013427197933197], [0.40820974111557007], [0.3767457604408264], [0.20149092376232147], [0.07500499486923218], [0.328750878572464], [0.31747785210609436], [0.24313530325889587]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4038843512535095], [0.44629916548728943], [0.008235457353293896], [0.14164923131465912], [0.18166889250278473], [0.179908886551857], [0.3791463375091553], [0.4934486150741577], [0.07347828894853592]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_5ed230dabd1be8b4889c6412889b4649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04966363683342934], [0.19880470633506775], [0.04978129267692566], [0.042689915746450424], [0.1670447289943695], [0.3255453109741211], [0.22168521583080292], [0.11604061722755432], [0.43528881669044495]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.22095853090286255], [0.2874982953071594], [0.0034736385568976402], [0.2015986442565918], [0.023722784593701363], [0.057638633996248245], [0.05427054315805435], [0.3369775414466858], [0.09985696524381638]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_9a160e5db521b1e3605d3c2e01550335(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41298598051071167], [0.48954424262046814], [0.2974533438682556], [0.04537849500775337], [0.4212323725223541], [0.14318227767944336], [0.2683485746383667], [0.40157246589660645], [0.34450793266296387]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.39238229393959045], [0.158734992146492], [0.4127102792263031], [0.41945791244506836], [0.29879409074783325], [0.013535123318433762], [0.1018265038728714], [0.21279259026050568], [0.030626868829131126]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_95f959d2068734b61c9187f7cf182653(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36636999249458313], [0.2677699625492096], [0.015402782708406448], [0.36943402886390686], [0.43435823917388916], [0.2168995440006256], [0.1261843889951706], [0.05905040726065636], [0.30280032753944397]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.09990614652633667], [0.32696646451950073], [0.2620046138763428], [0.3387252688407898], [0.2716210186481476], [0.1099492609500885], [0.29109078645706177], [0.030494073405861855], [0.40721428394317627]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_70327793ff638bfcae136c74b40881b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70327793ff638bfcae136c74b40881b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70327793ff638bfcae136c74b40881b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70327793ff638bfcae136c74b40881b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e9403775293db79ff3edd4f9816e864(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.040052223950624466, 0.026079408824443817, 0.223669171333313, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.33742555975914, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.09696357697248459], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8186bf40a33b503427fdc547afd28b4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1651548445224762, 0.4702686071395874, 0.36698785424232483, 0.0644494965672493, 0.4465831220149994, 0.07492171972990036], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1810450553894043, 0.2536962032318115, 0.2332039326429367, 0.22083275020122528, 0.2845509946346283, 0.10143007338047028], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_427c442e2c15f63fe31f9bebba9451da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.040052223950624466, 0.026079408824443817, 0.223669171333313, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3705940544605255, 0.04870932921767235, 0.07290997356176376, 0.22891433537006378, 0.06531837582588196, 0.44855570793151855], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_941843276467250c9de028bb6a29cf42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1651548445224762, 0.4702686071395874, 0.36698785424232483, 0.0644494965672493, 0.4465831220149994, 0.07492171972990036], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3314341604709625, 0.10902168601751328, 0.04308721795678139, 0.134864941239357, 0.4397190809249878, 0.21771910786628723], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_d584aa188dcdf5ec1ce51f9e50749342(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2057289332151413, 0.09391630440950394, 0.047374628484249115, 0.2254459410905838, 0.18349812924861908, 0.12224911153316498], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e763d8ebd5508c3fdfdf6acc5bed375a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95089378bd99bd9f812657fb508a008
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1810450553894043, 0.4702686071395874, 0.36698785424232483, 0.22083275020122528, 0.4465831220149994, 0.10143007338047028], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2081550508737564, 0.41452664136886597, 0.23729932308197021, 0.11266523599624634, 0.0390239953994751, 0.285847932100296], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_9c673d00482e80e36b1dc8a8dbd199e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c673d00482e80e36b1dc8a8dbd199e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c673d00482e80e36b1dc8a8dbd199e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c673d00482e80e36b1dc8a8dbd199e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_22663af05694d0fc91398ba9b7948e6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22663af05694d0fc91398ba9b7948e6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22663af05694d0fc91398ba9b7948e6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22663af05694d0fc91398ba9b7948e6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_5b8346acc90367e80a0919b98bcca0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2060057669878006]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.05126293748617172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_8a88f0105bc59254df47595d57c28f13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.26949211955070496]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3662005662918091]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_939b55f9f031baf0dad674069a88d657(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20535515248775482]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.08478929847478867]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_2ab0dd8f08b35e3896c052d4d33c2393(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.491769939661026]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.40578794479370117]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_7ba2af42cb15128d464f7bf0285f568a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3966229259967804], [0.2040838748216629], [0.2837878465652466], [0.1662617325782776], [0.08668506145477295], [0.3346942663192749]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17277514934539795], [0.4306377172470093], [0.21807396411895752], [0.11639563739299774], [0.04053834080696106], [0.47490552067756653]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_eb25e27ea97540770848167957a1f358(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49894431233406067], [0.3844832479953766], [0.21334876120090485], [0.25372934341430664], [0.006601519882678986], [0.32043617963790894]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.09358027577400208], [0.3427625000476837], [0.1698751002550125], [0.2064223736524582], [0.3752198815345764], [0.4434046745300293]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_d921db52310cd88555896973d9eb8438(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05556122213602066], [0.3532122075557709], [0.09650922566652298], [0.2863064706325531], [0.4242008626461029], [0.07322079688310623]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4361984133720398], [0.14439257979393005], [0.49514085054397583], [0.16213607788085938], [0.43578216433525085], [0.15172600746154785]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_37503a87ea2dbb6b57cbf62464919f0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04935036972165108], [0.401021808385849], [0.15970346331596375], [0.36091819405555725], [0.45432034134864807], [0.24458400905132294]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4443901479244232], [0.49830520153045654], [0.14623917639255524], [0.0019387512002140284], [0.4178217053413391], [0.1282878816127777]], dtype='float32').reshape([6, 1]),
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


    class TestPrimitiveOp_e364c81dde18161e23240968c757a4e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e364c81dde18161e23240968c757a4e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e364c81dde18161e23240968c757a4e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e364c81dde18161e23240968c757a4e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_2b97da7478ebfd831f7555f498ced0f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b97da7478ebfd831f7555f498ced0f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b97da7478ebfd831f7555f498ced0f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b97da7478ebfd831f7555f498ced0f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ede2cd79f421251a43e7074a71ef61c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ede2cd79f421251a43e7074a71ef61c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ede2cd79f421251a43e7074a71ef61c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ede2cd79f421251a43e7074a71ef61c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_95062009c83656883b46bd9e07b362b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08599822223186493], [0.21465438604354858], [0.023872841149568558], [0.1336366832256317], [0.12001485377550125]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.19767306745052338], [0.23302686214447021], [0.31559082865715027], [0.44974565505981445], [0.12936343252658844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_249fc5b8d2459701053bb60bec2908ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.31906890869140625], [0.02482573315501213], [0.439250111579895], [0.08620838075876236], [0.13948924839496613]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.22683300077915192], [0.04625708982348442], [0.1838442087173462], [0.45197582244873047], [0.287341833114624]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_9f2f95c778ee0c2679955ec3967407c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16905468702316284], [0.07408490031957626], [0.13317765295505524], [0.36288416385650635], [0.0633009746670723]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.4030136466026306], [0.48801735043525696], [0.401704341173172], [0.3282714784145355], [0.1866285353899002]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_03f13b7678906969890f4f4113ff3456(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15759168565273285], [0.09588633477687836], [0.37041568756103516], [0.4901740849018097], [0.2975691258907318]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2669719457626343], [0.3428363800048828], [0.49236804246902466], [0.149748757481575], [0.15492382645606995]], dtype='float32').reshape([5, 1]),
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


    class TestPrimitiveOp_f11ed117ceebc7288ae244b465a6fe8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f11ed117ceebc7288ae244b465a6fe8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f11ed117ceebc7288ae244b465a6fe8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f11ed117ceebc7288ae244b465a6fe8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4535f6f5d004dfa4158fae2237b9d6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4535f6f5d004dfa4158fae2237b9d6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4535f6f5d004dfa4158fae2237b9d6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4535f6f5d004dfa4158fae2237b9d6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f63a3d29ee47636d9e2d965422f7f02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f63a3d29ee47636d9e2d965422f7f02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f63a3d29ee47636d9e2d965422f7f02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f63a3d29ee47636d9e2d965422f7f02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_e5c9af3eeee6c0cc1a65f3be64a54dc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33240923285484314], [0.013569344766438007], [0.04725205898284912], [0.23816898465156555]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.40112707018852234], [0.16776657104492188], [0.05913810804486275], [0.45849156379699707]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_7f6f16d38d9cc9827132f25b073c3cc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.32093942165374756], [0.10365208238363266], [0.052328769117593765], [0.030598077923059464]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.21341432631015778], [0.16157600283622742], [0.17247040569782257], [0.42010706663131714]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_726e0ad432dcbdfe08000a7365713ca4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49404314160346985], [0.2501605749130249], [0.0006863751332275569], [0.08632545918226242]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.14522317051887512], [0.43382689356803894], [0.1448354572057724], [0.3873145282268524]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f9683f882f192217d7404013d0df090b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16857455670833588], [0.48723578453063965], [0.0007394644781015813], [0.12745331227779388]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.28212907910346985], [0.440621554851532], [0.13090020418167114], [0.48451167345046997]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_543c609df43f248bdab3771edecd94c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_543c609df43f248bdab3771edecd94c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_543c609df43f248bdab3771edecd94c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_543c609df43f248bdab3771edecd94c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_bf74cfa45fc9744350177d8e32be53ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf74cfa45fc9744350177d8e32be53ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf74cfa45fc9744350177d8e32be53ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf74cfa45fc9744350177d8e32be53ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88a5e4090695dc5292a01265289865cf
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
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