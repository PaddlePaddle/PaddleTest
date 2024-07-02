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
        paddle.seed(2024)
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
    class PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.floor_divide(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_492247920ace4675534661ab7578a46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(528, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3b6f9aa61e0c43bf38e32db45dc6ee2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(12, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_848331a6f67234905abce779ab993580(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(384, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4ec71836b894deddd1e70d9766ef501f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(20, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2be19112fb240ba4bbc41c7afb23bf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(2, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_97c88bc311ef820c88f075b6ef330930(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_237cb4871c4abb7421bc97fbe3de264f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(576, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_769b18fe43c18ffffd46d74204719461(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(96, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3b6f9aa61e0c43bf38e32db45dc6ee2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(12, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_fb4227ff93d1a1f14e17da0ac07517b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(960, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c3f4057a649047a651e56008cf548ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(2112, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_a0ef180a70645b9c16ec226baf133ab7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.floor_divide(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dedcc50e0d5539268f9882f2b84158b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0ef180a70645b9c16ec226baf133ab7
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int32').reshape([1]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_22d6b5e06d876591a83908f4c08f21f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0ef180a70645b9c16ec226baf133ab7
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_51b0e1169a628bf2b9ab80ffa4f2bd5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ade6d2164d64e5f37af9b3762f377f66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(240, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09f565cc0692f84292b9006e53208357(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(44, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_37bd20bc1023c04b3f0d0402449bc70b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0ef180a70645b9c16ec226baf133ab7
        def get_inputs(self):
            return [
                paddle.to_tensor([28], dtype='int32').reshape([1]),
                paddle.to_tensor(7, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c7672208fe4a5abd6059aa63352dfb75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0ef180a70645b9c16ec226baf133ab7
        def get_inputs(self):
            return [
                paddle.to_tensor([77], dtype='int32').reshape([1]),
                paddle.to_tensor(7, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_59901947df1de4031ec33026940a289b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(144, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_492247920ace4675534661ab7578a46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(528, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3b6f9aa61e0c43bf38e32db45dc6ee2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(12, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_848331a6f67234905abce779ab993580(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(384, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4ec71836b894deddd1e70d9766ef501f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(20, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2be19112fb240ba4bbc41c7afb23bf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(2, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_97c88bc311ef820c88f075b6ef330930(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_237cb4871c4abb7421bc97fbe3de264f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(576, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_769b18fe43c18ffffd46d74204719461(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(96, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3b6f9aa61e0c43bf38e32db45dc6ee2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(12, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_fb4227ff93d1a1f14e17da0ac07517b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(960, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4c3f4057a649047a651e56008cf548ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(2112, dtype='int32').reshape([]),
                paddle.to_tensor(96, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_4e937d94089d7483eab7e603545add15(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.floor_divide(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b82413bb0d06505e1c08997a5466cb8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e937d94089d7483eab7e603545add15
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int32').reshape([1]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ec095aad0c8a8974080197e5c6f31167(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e937d94089d7483eab7e603545add15
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_51b0e1169a628bf2b9ab80ffa4f2bd5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ade6d2164d64e5f37af9b3762f377f66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(240, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_09f565cc0692f84292b9006e53208357(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(44, dtype='int32').reshape([]),
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9c367759d3c0086b5da0cee6b54b0b69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e937d94089d7483eab7e603545add15
        def get_inputs(self):
            return [
                paddle.to_tensor([28], dtype='int32').reshape([1]),
                paddle.to_tensor(7, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_eb06dca43777c9b6e89de094d9f70c52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e937d94089d7483eab7e603545add15
        def get_inputs(self):
            return [
                paddle.to_tensor([77], dtype='int32').reshape([1]),
                paddle.to_tensor(7, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_59901947df1de4031ec33026940a289b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_825c0a881ec6e529e04b8ff3ca0e1fc3
        def get_inputs(self):
            return [
                paddle.to_tensor(144, dtype='int32').reshape([]),
                paddle.to_tensor(24, dtype='int32').reshape([]),
            ]


    

if __name__ == '__main__':
    unittest.main()