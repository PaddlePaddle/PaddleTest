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
    class PrimitiveOp_2f2ad784389d226e1cd01baa894e93b1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            input_0 = [11, 1, 1, 1]
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c4c4c8253d3c4791e7f1537fc592e341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f2ad784389d226e1cd01baa894e93b1
        def get_inputs(self):
            return [
                paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_84508bdb9a2161fd85d9612cd5761b11(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            input_0 = [43, 1, 1, 1]
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c3a28394ce74577f7e59f6e39d9092fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84508bdb9a2161fd85d9612cd5761b11
        def get_inputs(self):
            return [
                paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c3a28394ce74577f7e59f6e39d9092fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84508bdb9a2161fd85d9612cd5761b11
        def get_inputs(self):
            return [
                paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c4c4c8253d3c4791e7f1537fc592e341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f2ad784389d226e1cd01baa894e93b1
        def get_inputs(self):
            return [
                paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c4c4c8253d3c4791e7f1537fc592e341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f2ad784389d226e1cd01baa894e93b1
        def get_inputs(self):
            return [
                paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c3a28394ce74577f7e59f6e39d9092fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84508bdb9a2161fd85d9612cd5761b11
        def get_inputs(self):
            return [
                paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_6c869b4b516f236c6c2c972f07580707(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            input_0 = [1, 64, 1, 1]
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d664228ea3f2e85408b1351a6420a9e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c869b4b516f236c6c2c972f07580707
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 64, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_e218669fca6030132611885cb750b2b1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            input_0 = [1, 512, 1, 1]
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c3a28394ce74577f7e59f6e39d9092fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84508bdb9a2161fd85d9612cd5761b11
        def get_inputs(self):
            return [
                paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_8ff8ca16ce2130390dc67f286c102429(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            input_0 = [1, 192, 1, 1]
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_838779e1fb172ea3421a784211fd8303(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ff8ca16ce2130390dc67f286c102429
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 192, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d664228ea3f2e85408b1351a6420a9e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c869b4b516f236c6c2c972f07580707
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 64, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_01c154434dd70171018f638bac0dfb27(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            input_0 = [1, 256, 1, 1]
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ef4c4b1849bb88f9a0e1642548dbfdee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01c154434dd70171018f638bac0dfb27
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ef4c4b1849bb88f9a0e1642548dbfdee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01c154434dd70171018f638bac0dfb27
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_58aaa9006f79f6e82a86ac564eb8e3fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            input_0 = [1, 128, 1, 1]
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce6a77830b9c3d6287ed3f79c5c341d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58aaa9006f79f6e82a86ac564eb8e3fb
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c4c4c8253d3c4791e7f1537fc592e341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f2ad784389d226e1cd01baa894e93b1
        def get_inputs(self):
            return [
                paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_164490c6868bf4f049aa2393fa7e65a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            input_0 = [1, 2048, 1, 1]
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_678daa543471c59dc59760abf918dbc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_164490c6868bf4f049aa2393fa7e65a7
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 2048, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_678daa543471c59dc59760abf918dbc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_164490c6868bf4f049aa2393fa7e65a7
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 2048, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c4c4c8253d3c4791e7f1537fc592e341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f2ad784389d226e1cd01baa894e93b1
        def get_inputs(self):
            return [
                paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ce6a77830b9c3d6287ed3f79c5c341d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58aaa9006f79f6e82a86ac564eb8e3fb
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ef4c4b1849bb88f9a0e1642548dbfdee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01c154434dd70171018f638bac0dfb27
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c3a28394ce74577f7e59f6e39d9092fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84508bdb9a2161fd85d9612cd5761b11
        def get_inputs(self):
            return [
                paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ce6a77830b9c3d6287ed3f79c5c341d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58aaa9006f79f6e82a86ac564eb8e3fb
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c4c4c8253d3c4791e7f1537fc592e341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f2ad784389d226e1cd01baa894e93b1
        def get_inputs(self):
            return [
                paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c3a28394ce74577f7e59f6e39d9092fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84508bdb9a2161fd85d9612cd5761b11
        def get_inputs(self):
            return [
                paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c3a28394ce74577f7e59f6e39d9092fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84508bdb9a2161fd85d9612cd5761b11
        def get_inputs(self):
            return [
                paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c4c4c8253d3c4791e7f1537fc592e341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f2ad784389d226e1cd01baa894e93b1
        def get_inputs(self):
            return [
                paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c4c4c8253d3c4791e7f1537fc592e341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f2ad784389d226e1cd01baa894e93b1
        def get_inputs(self):
            return [
                paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c3a28394ce74577f7e59f6e39d9092fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84508bdb9a2161fd85d9612cd5761b11
        def get_inputs(self):
            return [
                paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d664228ea3f2e85408b1351a6420a9e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c869b4b516f236c6c2c972f07580707
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 64, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c3a28394ce74577f7e59f6e39d9092fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84508bdb9a2161fd85d9612cd5761b11
        def get_inputs(self):
            return [
                paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_838779e1fb172ea3421a784211fd8303(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ff8ca16ce2130390dc67f286c102429
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 192, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d664228ea3f2e85408b1351a6420a9e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c869b4b516f236c6c2c972f07580707
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 64, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ef4c4b1849bb88f9a0e1642548dbfdee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01c154434dd70171018f638bac0dfb27
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ef4c4b1849bb88f9a0e1642548dbfdee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01c154434dd70171018f638bac0dfb27
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ce6a77830b9c3d6287ed3f79c5c341d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58aaa9006f79f6e82a86ac564eb8e3fb
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c4c4c8253d3c4791e7f1537fc592e341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f2ad784389d226e1cd01baa894e93b1
        def get_inputs(self):
            return [
                paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_678daa543471c59dc59760abf918dbc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_164490c6868bf4f049aa2393fa7e65a7
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 2048, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_678daa543471c59dc59760abf918dbc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_164490c6868bf4f049aa2393fa7e65a7
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 2048, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c4c4c8253d3c4791e7f1537fc592e341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f2ad784389d226e1cd01baa894e93b1
        def get_inputs(self):
            return [
                paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ce6a77830b9c3d6287ed3f79c5c341d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58aaa9006f79f6e82a86ac564eb8e3fb
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ef4c4b1849bb88f9a0e1642548dbfdee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01c154434dd70171018f638bac0dfb27
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c3a28394ce74577f7e59f6e39d9092fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84508bdb9a2161fd85d9612cd5761b11
        def get_inputs(self):
            return [
                paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6940531dbc38f6e759cba968d98331b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e218669fca6030132611885cb750b2b1
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ce6a77830b9c3d6287ed3f79c5c341d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58aaa9006f79f6e82a86ac564eb8e3fb
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_87e0e6a6a33595af437ec36c0d1671a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            input_0 = [11, 1, 1, 1]
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_14d840247fac7ff03ba133b3c986eb44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87e0e6a6a33595af437ec36c0d1671a2
        def get_inputs(self):
            return [
                paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_b1aec51258dd61970bca32ded1cc76bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            input_0 = [43, 1, 1, 1]
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_858e7fbf20077a5d4aedd6cdde3b1677(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1aec51258dd61970bca32ded1cc76bf
        def get_inputs(self):
            return [
                paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_858e7fbf20077a5d4aedd6cdde3b1677(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1aec51258dd61970bca32ded1cc76bf
        def get_inputs(self):
            return [
                paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_14d840247fac7ff03ba133b3c986eb44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87e0e6a6a33595af437ec36c0d1671a2
        def get_inputs(self):
            return [
                paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_14d840247fac7ff03ba133b3c986eb44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87e0e6a6a33595af437ec36c0d1671a2
        def get_inputs(self):
            return [
                paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_858e7fbf20077a5d4aedd6cdde3b1677(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1aec51258dd61970bca32ded1cc76bf
        def get_inputs(self):
            return [
                paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_4797c3b14c6d3c1bcf1e29847621b614(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            input_0 = [1, 64, 1, 1]
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_17d4009d99b5851a42363abaa6dd65d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4797c3b14c6d3c1bcf1e29847621b614
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 64, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_1227f3e38e7d7a95b70d66ddcde5c032(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            input_0 = [1, 512, 1, 1]
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f6d08f982455b037fcf9cd2e2655a647(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1227f3e38e7d7a95b70d66ddcde5c032
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_858e7fbf20077a5d4aedd6cdde3b1677(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1aec51258dd61970bca32ded1cc76bf
        def get_inputs(self):
            return [
                paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_70658a0f47066e92d3a31eead3e7d7b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            input_0 = [1, 192, 1, 1]
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6fea378740283141d6d4ced49d952969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70658a0f47066e92d3a31eead3e7d7b8
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 192, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_17d4009d99b5851a42363abaa6dd65d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4797c3b14c6d3c1bcf1e29847621b614
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 64, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_cd3af6c9d002e49c962cd4b99c9228e2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            input_0 = [1, 256, 1, 1]
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9284131361280a9371ba4a185d702ec9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd3af6c9d002e49c962cd4b99c9228e2
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9284131361280a9371ba4a185d702ec9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd3af6c9d002e49c962cd4b99c9228e2
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_571666085f7d83bc9f2d324fe3424e43(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            input_0 = [1, 128, 1, 1]
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c93ee5cf9c9c697a8e5169619e2e6b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_571666085f7d83bc9f2d324fe3424e43
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_14d840247fac7ff03ba133b3c986eb44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87e0e6a6a33595af437ec36c0d1671a2
        def get_inputs(self):
            return [
                paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f6d08f982455b037fcf9cd2e2655a647(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1227f3e38e7d7a95b70d66ddcde5c032
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f6d08f982455b037fcf9cd2e2655a647(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1227f3e38e7d7a95b70d66ddcde5c032
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f6d08f982455b037fcf9cd2e2655a647(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1227f3e38e7d7a95b70d66ddcde5c032
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_c1df68957563c531cba4587531556556(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            input_0 = [1, 2048, 1, 1]
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c2b576c8adfd1cf29e29f92166dc2bf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1df68957563c531cba4587531556556
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 2048, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f6d08f982455b037fcf9cd2e2655a647(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1227f3e38e7d7a95b70d66ddcde5c032
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f6d08f982455b037fcf9cd2e2655a647(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1227f3e38e7d7a95b70d66ddcde5c032
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f6d08f982455b037fcf9cd2e2655a647(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1227f3e38e7d7a95b70d66ddcde5c032
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c2b576c8adfd1cf29e29f92166dc2bf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1df68957563c531cba4587531556556
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 2048, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_14d840247fac7ff03ba133b3c986eb44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87e0e6a6a33595af437ec36c0d1671a2
        def get_inputs(self):
            return [
                paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c93ee5cf9c9c697a8e5169619e2e6b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_571666085f7d83bc9f2d324fe3424e43
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9284131361280a9371ba4a185d702ec9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd3af6c9d002e49c962cd4b99c9228e2
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_858e7fbf20077a5d4aedd6cdde3b1677(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1aec51258dd61970bca32ded1cc76bf
        def get_inputs(self):
            return [
                paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f6d08f982455b037fcf9cd2e2655a647(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1227f3e38e7d7a95b70d66ddcde5c032
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f6d08f982455b037fcf9cd2e2655a647(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1227f3e38e7d7a95b70d66ddcde5c032
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c93ee5cf9c9c697a8e5169619e2e6b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_571666085f7d83bc9f2d324fe3424e43
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()