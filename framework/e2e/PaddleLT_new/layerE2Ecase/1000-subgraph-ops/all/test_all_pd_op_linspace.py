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
    class PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.linspace(input_0, input_1, input_2, paddle.float32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_45fe07b6467668ecf12ef3c092a89515(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([36], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_df57fad634ea5938154ca0ebedc939bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([24], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_db778747ed292c05b7242a8dcdb8c61e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([38], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7f131b9b0f91d11fcacc663d9315d5e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([25], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2db8db5350541954043e3f9826557b79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([30], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6b99719c48f08c71e031e67bdd754054(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7f131b9b0f91d11fcacc663d9315d5e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([25], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_650f7564c73662ac3210305a54af2d0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_45fe07b6467668ecf12ef3c092a89515(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([36], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_df57fad634ea5938154ca0ebedc939bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([24], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_db778747ed292c05b7242a8dcdb8c61e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([38], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7f131b9b0f91d11fcacc663d9315d5e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([25], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2db8db5350541954043e3f9826557b79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([30], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6b99719c48f08c71e031e67bdd754054(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7f131b9b0f91d11fcacc663d9315d5e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([25], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_650f7564c73662ac3210305a54af2d0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ce99b32d2d7386ff11ae57dbf68df6c
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_91d132c9ea5e71bcd577c690ecd4a2a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.linspace(input_0, input_1, input_2, paddle.float32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e45fe4a14ed6badc2bf699f20b1216bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91d132c9ea5e71bcd577c690ecd4a2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([36], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4906e726aad72e86d960913a34b70199(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91d132c9ea5e71bcd577c690ecd4a2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([24], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_64e5220fe1a878d212da05125159f316(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91d132c9ea5e71bcd577c690ecd4a2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([38], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0918cd7974d28f6530b9fb9d944cd14a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91d132c9ea5e71bcd577c690ecd4a2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([25], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_94de4597d267cb857901f665a3ef180b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91d132c9ea5e71bcd577c690ecd4a2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([30], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_82fc27de453475a2282713ee805fbd01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91d132c9ea5e71bcd577c690ecd4a2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0918cd7974d28f6530b9fb9d944cd14a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91d132c9ea5e71bcd577c690ecd4a2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([25], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6f0ada14b9f142426f5ff0d4b5e2b11d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91d132c9ea5e71bcd577c690ecd4a2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15], dtype='int32').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()