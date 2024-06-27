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
    class PrimitiveOp_fe6c05ff08fc00dc2c7ab9f91aa4537d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bce_loss(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
                paddle.static.InputSpec(shape=[3800, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_22c8771f05861abefa2d260f540ad923(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6c05ff08fc00dc2c7ab9f91aa4537d
        def get_inputs(self):
            return [
                paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b9af8ee3937054d4f248c67f4eae25e9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bce_loss(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7bb6ac446149a62ccc8b980bf9b2d7b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9af8ee3937054d4f248c67f4eae25e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cfbb8b92a18ec21a93453009f703abaa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bce_loss(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
                paddle.static.InputSpec(shape=[15200, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd2758bdfeded01919818f10bff11c26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfbb8b92a18ec21a93453009f703abaa
        def get_inputs(self):
            return [
                paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f4befb8d62354e9535f68507792a682d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bce_loss(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
                paddle.static.InputSpec(shape=[70, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eb7f749dff5395690c2526f271b3d952(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4befb8d62354e9535f68507792a682d
        def get_inputs(self):
            return [
                paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ee78adbf61c13d71d1b43b599e818251(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bce_loss(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
                paddle.static.InputSpec(shape=[247, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_918e7ace56334b741f61f5995f6b3f25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee78adbf61c13d71d1b43b599e818251
        def get_inputs(self):
            return [
                paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e71ef775d9db73e9cad5ddafd830a4f6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bce_loss(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
                paddle.static.InputSpec(shape=[950, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3c7ae55fa5864727974bac1e49c39295(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e71ef775d9db73e9cad5ddafd830a4f6
        def get_inputs(self):
            return [
                paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e7597a4f85c39b89e67202bd0262fc8d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bce_loss(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3800, 81], dtype='float32'),
                paddle.static.InputSpec(shape=[3800, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d2f91886c402c388699fbebfef292f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e7597a4f85c39b89e67202bd0262fc8d
        def get_inputs(self):
            return [
                paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d156fe250ac0b5bf28f408d4dc3f2d22(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bce_loss(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 21824, 15], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 21824, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_372fe0ba2f244a6eca208c6a944489eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d156fe250ac0b5bf28f408d4dc3f2d22
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_972ee48a9999a1cd674156dae56da2ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bce_loss(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[15200, 81], dtype='float32'),
                paddle.static.InputSpec(shape=[15200, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_54f9875b4c99227b0f23aef37beee1e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_972ee48a9999a1cd674156dae56da2ef
        def get_inputs(self):
            return [
                paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d90fe1e9e3e9360f9366bc1f2cd08073(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bce_loss(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[70, 81], dtype='float32'),
                paddle.static.InputSpec(shape=[70, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b26a6b200e788733bc049d0d849b9c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d90fe1e9e3e9360f9366bc1f2cd08073
        def get_inputs(self):
            return [
                paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d50f582d882c74c4dee75c0e34b6b312(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bce_loss(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[247, 81], dtype='float32'),
                paddle.static.InputSpec(shape=[247, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4f6901982086f60b932d320dbdafe5fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d50f582d882c74c4dee75c0e34b6b312
        def get_inputs(self):
            return [
                paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a69797702f1d9ff49d26ed32f942ec38(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bce_loss(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[950, 81], dtype='float32'),
                paddle.static.InputSpec(shape=[950, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c6e3021a0411fcbb227a2e64ded5f5c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a69797702f1d9ff49d26ed32f942ec38
        def get_inputs(self):
            return [
                paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_869befa69546e3c66f6d131e7651a0d2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bce_loss(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f0824c581a5ef08de726c925d675ae7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_869befa69546e3c66f6d131e7651a0d2
        def get_inputs(self):
            return [
                paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bb6ac446149a62ccc8b980bf9b2d7b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9af8ee3937054d4f248c67f4eae25e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f07253e5b3b26c5cea95b0d6f95438a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_869befa69546e3c66f6d131e7651a0d2
        def get_inputs(self):
            return [
                paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09adfe454a27ce7e200b82415e9f8cfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_869befa69546e3c66f6d131e7651a0d2
        def get_inputs(self):
            return [
                paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f826a9f0d7cf66943dd73e92025ebf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_869befa69546e3c66f6d131e7651a0d2
        def get_inputs(self):
            return [
                paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b0dbb165a88134eb2237a346d6d9696(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_869befa69546e3c66f6d131e7651a0d2
        def get_inputs(self):
            return [
                paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()