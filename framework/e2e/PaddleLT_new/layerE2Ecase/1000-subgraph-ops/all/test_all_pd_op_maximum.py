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
    class PrimitiveOp_44bd0c7dece725587123f4604e106d44(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8aff6f84a34d7a0a60d6f60b64f2a660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8aff6f84a34d7a0a60d6f60b64f2a660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9d2be0d7f27e44ab009170c9dc53aad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9d2be0d7f27e44ab009170c9dc53aad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb19a46cb4a00d0fdcc83e415ae800cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb19a46cb4a00d0fdcc83e415ae800cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1fd3aae887de9bc2d1245297218f5f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1fd3aae887de9bc2d1245297218f5f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04481b9bfaff5c8ada76d7111dc5b36d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04481b9bfaff5c8ada76d7111dc5b36d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94679c7a8653cae3932890b0e15e275c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94679c7a8653cae3932890b0e15e275c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8aff6f84a34d7a0a60d6f60b64f2a660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8aff6f84a34d7a0a60d6f60b64f2a660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de0146d98124e8f73ad587a92ee850e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de0146d98124e8f73ad587a92ee850e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f9a60539947be8a12d4b323cde1640f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f9a60539947be8a12d4b323cde1640f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c1e61dd8574af3091708fc915d30beda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1e61dd8574af3091708fc915d30beda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1e61dd8574af3091708fc915d30beda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1e61dd8574af3091708fc915d30beda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7f3d7d11be6bef005875a691d131ce3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7f3d7d11be6bef005875a691d131ce3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87637564cb35f755c65faf45c81c4479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87637564cb35f755c65faf45c81c4479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbf43afc3c9144784bf2c425282bad97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbf43afc3c9144784bf2c425282bad97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02bd2629126b35cdecb9b0fc7fea30da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02bd2629126b35cdecb9b0fc7fea30da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_05bf30e0a86497370f67aa4f126e947c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3abf5191c5db634641d6bfcebe3dcbc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14307871460914612], [0.19619788229465485], [0.1863391399383545], [0.2124066799879074], [0.035387687385082245], [0.1282057762145996], [0.4028562605381012], [0.007067443337291479], [0.31240081787109375]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.38795778155326843], [0.4817730784416199], [0.017573527991771698], [0.44741085171699524], [0.29538026452064514], [0.3727307915687561], [0.023896757513284683], [0.3053695261478424], [0.20216022431850433]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_027d407b678a0a1e24b1b0d46e1f8722(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14178644120693207], [0.06213818117976189], [0.3997913599014282], [0.11639508605003357], [0.015070164576172829], [0.12873531877994537], [0.05166096240282059], [0.269934743642807], [0.09819310158491135]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.49473410844802856], [0.3323400914669037], [0.2790012061595917], [0.4925849139690399], [0.17630773782730103], [0.10014522075653076], [0.08800354599952698], [0.43823152780532837], [0.4259171187877655]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_549c96d97a50a2670d8e2cb9ca7d7eb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13368675112724304], [0.39139324426651], [0.3978815972805023], [0.2982294261455536], [0.2754661738872528], [0.3869280219078064], [0.10337948054075241], [0.465537965297699], [0.4861213266849518]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.2243945300579071], [0.3421638607978821], [0.4742037057876587], [0.15133216977119446], [0.22187696397304535], [0.2794041037559509], [0.01730276271700859], [0.08600067347288132], [0.0719427838921547]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_90c149000876c4f8d008d6986ca9c410(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16374197602272034], [0.22206810116767883], [0.18589580059051514], [0.26141688227653503], [0.006397350691258907], [0.38319242000579834], [0.3071695864200592], [0.40571993589401245], [0.43421873450279236]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4538259506225586], [0.18657687306404114], [0.13615421950817108], [0.37509968876838684], [0.4772617220878601], [0.39100539684295654], [0.4677458703517914], [0.004688146524131298], [0.23858243227005005]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_dbfe32a5ff890b4c2fb9b939f54d3ec8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dbfe32a5ff890b4c2fb9b939f54d3ec8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dbfe32a5ff890b4c2fb9b939f54d3ec8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dbfe32a5ff890b4c2fb9b939f54d3ec8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_101f7678f25d83eadc2aab2cee819321(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0044a42cdec3136931ecea904724c2b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101f7678f25d83eadc2aab2cee819321
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2341037541627884, 0.3048173189163208, 0.39980247616767883, 0.10136908292770386, 0.32081952691078186, 0.11360201984643936], dtype='float32').reshape([6]),
                paddle.to_tensor([0.45227816700935364, 0.2502661347389221, 0.16833464801311493, 0.15319602191448212, 0.18911296129226685, 0.1342129111289978], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_823d5569a1a5129af329df486f60f7d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101f7678f25d83eadc2aab2cee819321
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3236333429813385, 0.01610579900443554, 0.14992783963680267, 0.4712778925895691, 0.4077642858028412, 0.023976648226380348], dtype='float32').reshape([6]),
                paddle.to_tensor([0.27197179198265076, 0.4429328143596649, 0.17026685178279877, 0.08001313358545303, 0.23915867507457733, 0.05345135182142258], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_bfc22a86530bc99f45c9a1ba06060aa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101f7678f25d83eadc2aab2cee819321
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2341037541627884, 0.3048173189163208, 0.39980247616767883, 0.10136908292770386, 0.32081952691078186, 0.11360201984643936], dtype='float32').reshape([6]),
                paddle.to_tensor([0.49831607937812805, 0.37963253259658813, 0.22442752122879028, 0.4289640784263611, 0.301647424697876, 0.0385211743414402], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_02520dd41ba454446871bac1c3b30493(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101f7678f25d83eadc2aab2cee819321
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3236333429813385, 0.01610579900443554, 0.14992783963680267, 0.4712778925895691, 0.4077642858028412, 0.023976648226380348], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3058600425720215, 0.14679817855358124, 0.3744381070137024, 0.0002531889476813376, 0.2294905185699463, 0.37590494751930237], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c854ced62b0ba16cff251671830d0491(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101f7678f25d83eadc2aab2cee819321
        def get_inputs(self):
            return [
                paddle.to_tensor([0.45227816700935364, 0.3048173189163208, 0.39980247616767883, 0.15319602191448212, 0.32081952691078186, 0.1342129111289978], dtype='float32').reshape([6]),
                paddle.to_tensor([0.22168461978435516, 0.48501694202423096, 0.36503466963768005, 0.4213145971298218, 0.4012173116207123, 0.14370423555374146], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_21fe36b506476905a5c2da8f905e41d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101f7678f25d83eadc2aab2cee819321
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3236333429813385, 0.4429328143596649, 0.17026685178279877, 0.4712778925895691, 0.4077642858028412, 0.05345135182142258], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3964419662952423, 0.30897533893585205, 0.40568625926971436, 0.10646888613700867, 0.4039801359176636, 0.40599721670150757], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_7dfdb7f5765314d0f6686d0e15e4af14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7dfdb7f5765314d0f6686d0e15e4af14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7dfdb7f5765314d0f6686d0e15e4af14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7dfdb7f5765314d0f6686d0e15e4af14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_99175e039626b291c560242619e0f1bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e599c52f3e820352ad175f1194f0900(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99175e039626b291c560242619e0f1bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1.000000013351432e-10], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c1fd3aae887de9bc2d1245297218f5f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1fd3aae887de9bc2d1245297218f5f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87637564cb35f755c65faf45c81c4479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87637564cb35f755c65faf45c81c4479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9da01e3b1bfb303b5a115ca287475c1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9da01e3b1bfb303b5a115ca287475c1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9da01e3b1bfb303b5a115ca287475c1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9da01e3b1bfb303b5a115ca287475c1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94679c7a8653cae3932890b0e15e275c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94679c7a8653cae3932890b0e15e275c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9d2be0d7f27e44ab009170c9dc53aad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9d2be0d7f27e44ab009170c9dc53aad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73d92fa0cf2cd56eb13968436b3e74d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.178619846701622]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.20313051342964172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ad6bbf3a067c43c836d96ca8aecfd87e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15484283864498138]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.004780620336532593]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_2916c9c5c01f26a00e722481ce70ef44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.27537572383880615]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.31011876463890076]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_8219f1b3815f08f4da10a4f0ae73e6c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.19937725365161896]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.14600974321365356]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_8641c44d1fdc5f5f6fde07a587682ddb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3637081980705261], [0.39640137553215027], [0.31833428144454956], [0.4412391483783722], [0.11331885308027267], [0.4164228141307831]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.269661545753479], [0.24576835334300995], [0.09308407455682755], [0.3879702687263489], [0.43500569462776184], [0.14424768090248108]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_286d30d54c383de0f3d6d38ef22979b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2915404736995697], [0.06591049581766129], [0.19010786712169647], [0.17498598992824554], [0.26003992557525635], [0.24949562549591064]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.25507786870002747], [0.1776416003704071], [0.0561361089348793], [0.35384827852249146], [0.1015116423368454], [0.23805175721645355]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_b9616e9bd6abd793fecaacac85e6c423(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28181594610214233], [0.256033331155777], [0.3946605324745178], [0.4794350862503052], [0.0947040542960167], [0.32221317291259766]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0779203251004219], [0.28568974137306213], [0.2259289026260376], [0.08137910068035126], [0.37925606966018677], [0.45194125175476074]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_a23e13b221b539b7eae7cf618fdd123a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4542856514453888], [0.32371410727500916], [0.47006481885910034], [0.24301113188266754], [0.31819093227386475], [0.3337242603302002]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4113561809062958], [0.38559842109680176], [0.34662380814552307], [0.25977855920791626], [0.09847179055213928], [0.15297983586788177]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_e7f3d7d11be6bef005875a691d131ce3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7f3d7d11be6bef005875a691d131ce3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb19a46cb4a00d0fdcc83e415ae800cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb19a46cb4a00d0fdcc83e415ae800cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66cc2b47056230fa725832f70d8487d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66cc2b47056230fa725832f70d8487d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66cc2b47056230fa725832f70d8487d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66cc2b47056230fa725832f70d8487d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53fa46e838ce4df94a0de0be6d560cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53fa46e838ce4df94a0de0be6d560cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b230318d1ac6b96e244f39f5ebf713c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b230318d1ac6b96e244f39f5ebf713c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b230318d1ac6b96e244f39f5ebf713c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b230318d1ac6b96e244f39f5ebf713c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18dc2d0d7c487f6a6220e3efd2373627(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18dc2d0d7c487f6a6220e3efd2373627(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18dc2d0d7c487f6a6220e3efd2373627(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18dc2d0d7c487f6a6220e3efd2373627(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04481b9bfaff5c8ada76d7111dc5b36d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04481b9bfaff5c8ada76d7111dc5b36d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbf43afc3c9144784bf2c425282bad97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbf43afc3c9144784bf2c425282bad97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17606de4874b9b0799f2069684810597(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25348156690597534], [0.1419389396905899], [0.21522848308086395], [0.18140600621700287], [0.35213175415992737]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1649070680141449], [0.39336472749710083], [0.47650161385536194], [0.22819465398788452], [0.16766460239887238]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_0a3de64a88b145cf53efc168b8c2c138(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.47471386194229126], [0.33389967679977417], [0.40402311086654663], [0.4485560953617096], [0.46584224700927734]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.21730045974254608], [0.35416045784950256], [0.32481035590171814], [0.011503858491778374], [0.4248878061771393]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_585419f76490241eaa26dc41626ed7c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18775731325149536], [0.12440189719200134], [0.05388212949037552], [0.37235766649246216], [0.08130541443824768]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.13528534770011902], [0.3804410994052887], [0.13431361317634583], [0.32307666540145874], [0.13232417404651642]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_a340aff25ead7c12a739411ae888407c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.21230792999267578], [0.0743480697274208], [0.0162972342222929], [0.3133012354373932], [0.06640253961086273]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.4135350286960602], [0.022249840199947357], [0.3079543709754944], [0.28616321086883545], [0.11313261836767197]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_2f9a60539947be8a12d4b323cde1640f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f9a60539947be8a12d4b323cde1640f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02bd2629126b35cdecb9b0fc7fea30da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02bd2629126b35cdecb9b0fc7fea30da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2a3a4d3dde8652ea816d52cbac75101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2a3a4d3dde8652ea816d52cbac75101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_760459256eb106b18218f8d3ae8090fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_760459256eb106b18218f8d3ae8090fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_760459256eb106b18218f8d3ae8090fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_760459256eb106b18218f8d3ae8090fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89997cef3610b8bf9c64582d7c90387a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89997cef3610b8bf9c64582d7c90387a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89997cef3610b8bf9c64582d7c90387a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89997cef3610b8bf9c64582d7c90387a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a59dc4058a40bde3a2d595b08662e94d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a59dc4058a40bde3a2d595b08662e94d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a59dc4058a40bde3a2d595b08662e94d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a59dc4058a40bde3a2d595b08662e94d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53fa46e838ce4df94a0de0be6d560cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53fa46e838ce4df94a0de0be6d560cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b66657045ea52b2a556d191be82db1e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.43446049094200134], [0.3114164471626282], [0.21863189339637756], [0.22740739583969116]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.05875740945339203], [0.3892061412334442], [0.20329052209854126], [0.3478715717792511]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_6658cd23850aaf776f1e34146854eeb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28952881693840027], [0.0999356284737587], [0.3750646412372589], [0.1684504747390747]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2514371871948242], [0.428699254989624], [0.0340176485478878], [0.16604164242744446]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f6812f4f9efeda0dc985d35fe13c6761(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1857021450996399], [0.232812762260437], [0.44483304023742676], [0.43192264437675476]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.37367141246795654], [0.26348745822906494], [0.22833964228630066], [0.3062015175819397]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_6eed9f9e94fbfee95574b0ecb9c87412(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28646060824394226], [0.2347613275051117], [0.07614689320325851], [0.21912746131420135]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.3861541450023651], [0.4490680992603302], [0.4151049554347992], [0.03166484087705612]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_260cbf3cb5bf17f0ad112f5e43ab7012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_260cbf3cb5bf17f0ad112f5e43ab7012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_260cbf3cb5bf17f0ad112f5e43ab7012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_260cbf3cb5bf17f0ad112f5e43ab7012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2a3a4d3dde8652ea816d52cbac75101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2a3a4d3dde8652ea816d52cbac75101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de0146d98124e8f73ad587a92ee850e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de0146d98124e8f73ad587a92ee850e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c31933cd677bc48ccc1f9ff263f5c915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c31933cd677bc48ccc1f9ff263f5c915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_213e6c01f71902cba9e967aa12876cc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_213e6c01f71902cba9e967aa12876cc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_213e6c01f71902cba9e967aa12876cc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_213e6c01f71902cba9e967aa12876cc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28f27ae47d08f73f0926dcde2f6bdac
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c31933cd677bc48ccc1f9ff263f5c915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c31933cd677bc48ccc1f9ff263f5c915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8aff6f84a34d7a0a60d6f60b64f2a660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8aff6f84a34d7a0a60d6f60b64f2a660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9d2be0d7f27e44ab009170c9dc53aad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9d2be0d7f27e44ab009170c9dc53aad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb19a46cb4a00d0fdcc83e415ae800cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb19a46cb4a00d0fdcc83e415ae800cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1fd3aae887de9bc2d1245297218f5f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1fd3aae887de9bc2d1245297218f5f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04481b9bfaff5c8ada76d7111dc5b36d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04481b9bfaff5c8ada76d7111dc5b36d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94679c7a8653cae3932890b0e15e275c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94679c7a8653cae3932890b0e15e275c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8aff6f84a34d7a0a60d6f60b64f2a660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8aff6f84a34d7a0a60d6f60b64f2a660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de0146d98124e8f73ad587a92ee850e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de0146d98124e8f73ad587a92ee850e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f9a60539947be8a12d4b323cde1640f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f9a60539947be8a12d4b323cde1640f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ebec1aa4513652f593d705e560a9b32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ebec1aa4513652f593d705e560a9b32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ebec1aa4513652f593d705e560a9b32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ebec1aa4513652f593d705e560a9b32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7f3d7d11be6bef005875a691d131ce3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7f3d7d11be6bef005875a691d131ce3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87637564cb35f755c65faf45c81c4479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87637564cb35f755c65faf45c81c4479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbf43afc3c9144784bf2c425282bad97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbf43afc3c9144784bf2c425282bad97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02bd2629126b35cdecb9b0fc7fea30da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02bd2629126b35cdecb9b0fc7fea30da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3abf5191c5db634641d6bfcebe3dcbc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14307871460914612], [0.19619788229465485], [0.1863391399383545], [0.2124066799879074], [0.035387687385082245], [0.1282057762145996], [0.4028562605381012], [0.007067443337291479], [0.31240081787109375]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.38795778155326843], [0.4817730784416199], [0.017573527991771698], [0.44741085171699524], [0.29538026452064514], [0.3727307915687561], [0.023896757513284683], [0.3053695261478424], [0.20216022431850433]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_027d407b678a0a1e24b1b0d46e1f8722(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14178644120693207], [0.06213818117976189], [0.3997913599014282], [0.11639508605003357], [0.015070164576172829], [0.12873531877994537], [0.05166096240282059], [0.269934743642807], [0.09819310158491135]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.49473410844802856], [0.3323400914669037], [0.2790012061595917], [0.4925849139690399], [0.17630773782730103], [0.10014522075653076], [0.08800354599952698], [0.43823152780532837], [0.4259171187877655]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_549c96d97a50a2670d8e2cb9ca7d7eb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13368675112724304], [0.39139324426651], [0.3978815972805023], [0.2982294261455536], [0.2754661738872528], [0.3869280219078064], [0.10337948054075241], [0.465537965297699], [0.4861213266849518]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.2243945300579071], [0.3421638607978821], [0.4742037057876587], [0.15133216977119446], [0.22187696397304535], [0.2794041037559509], [0.01730276271700859], [0.08600067347288132], [0.0719427838921547]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_90c149000876c4f8d008d6986ca9c410(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16374197602272034], [0.22206810116767883], [0.18589580059051514], [0.26141688227653503], [0.006397350691258907], [0.38319242000579834], [0.3071695864200592], [0.40571993589401245], [0.43421873450279236]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4538259506225586], [0.18657687306404114], [0.13615421950817108], [0.37509968876838684], [0.4772617220878601], [0.39100539684295654], [0.4677458703517914], [0.004688146524131298], [0.23858243227005005]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_eba67c224784b223ec52cdf94a4d3c79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eba67c224784b223ec52cdf94a4d3c79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eba67c224784b223ec52cdf94a4d3c79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eba67c224784b223ec52cdf94a4d3c79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0044a42cdec3136931ecea904724c2b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101f7678f25d83eadc2aab2cee819321
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2341037541627884, 0.3048173189163208, 0.39980247616767883, 0.10136908292770386, 0.32081952691078186, 0.11360201984643936], dtype='float32').reshape([6]),
                paddle.to_tensor([0.45227816700935364, 0.2502661347389221, 0.16833464801311493, 0.15319602191448212, 0.18911296129226685, 0.1342129111289978], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_823d5569a1a5129af329df486f60f7d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101f7678f25d83eadc2aab2cee819321
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3236333429813385, 0.01610579900443554, 0.14992783963680267, 0.4712778925895691, 0.4077642858028412, 0.023976648226380348], dtype='float32').reshape([6]),
                paddle.to_tensor([0.27197179198265076, 0.4429328143596649, 0.17026685178279877, 0.08001313358545303, 0.23915867507457733, 0.05345135182142258], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_bfc22a86530bc99f45c9a1ba06060aa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101f7678f25d83eadc2aab2cee819321
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2341037541627884, 0.3048173189163208, 0.39980247616767883, 0.10136908292770386, 0.32081952691078186, 0.11360201984643936], dtype='float32').reshape([6]),
                paddle.to_tensor([0.49831607937812805, 0.37963253259658813, 0.22442752122879028, 0.4289640784263611, 0.301647424697876, 0.0385211743414402], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_02520dd41ba454446871bac1c3b30493(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101f7678f25d83eadc2aab2cee819321
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3236333429813385, 0.01610579900443554, 0.14992783963680267, 0.4712778925895691, 0.4077642858028412, 0.023976648226380348], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3058600425720215, 0.14679817855358124, 0.3744381070137024, 0.0002531889476813376, 0.2294905185699463, 0.37590494751930237], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c854ced62b0ba16cff251671830d0491(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101f7678f25d83eadc2aab2cee819321
        def get_inputs(self):
            return [
                paddle.to_tensor([0.45227816700935364, 0.3048173189163208, 0.39980247616767883, 0.15319602191448212, 0.32081952691078186, 0.1342129111289978], dtype='float32').reshape([6]),
                paddle.to_tensor([0.22168461978435516, 0.48501694202423096, 0.36503466963768005, 0.4213145971298218, 0.4012173116207123, 0.14370423555374146], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_21fe36b506476905a5c2da8f905e41d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101f7678f25d83eadc2aab2cee819321
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3236333429813385, 0.4429328143596649, 0.17026685178279877, 0.4712778925895691, 0.4077642858028412, 0.05345135182142258], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3964419662952423, 0.30897533893585205, 0.40568625926971436, 0.10646888613700867, 0.4039801359176636, 0.40599721670150757], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_08bb6277f84b48456101d2c921f685c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08bb6277f84b48456101d2c921f685c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08bb6277f84b48456101d2c921f685c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08bb6277f84b48456101d2c921f685c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0fc647630f2ea7ae5a02e69506ce414c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.maximum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e3624bd491c1af30e7f3f9fcf788be7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fc647630f2ea7ae5a02e69506ce414c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1.000000013351432e-10], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c1fd3aae887de9bc2d1245297218f5f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1fd3aae887de9bc2d1245297218f5f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87637564cb35f755c65faf45c81c4479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87637564cb35f755c65faf45c81c4479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17cfabcc15ab25355cd077decce4a2d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17cfabcc15ab25355cd077decce4a2d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17cfabcc15ab25355cd077decce4a2d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17cfabcc15ab25355cd077decce4a2d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94679c7a8653cae3932890b0e15e275c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94679c7a8653cae3932890b0e15e275c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9d2be0d7f27e44ab009170c9dc53aad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9d2be0d7f27e44ab009170c9dc53aad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73d92fa0cf2cd56eb13968436b3e74d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.178619846701622]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.20313051342964172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ad6bbf3a067c43c836d96ca8aecfd87e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15484283864498138]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.004780620336532593]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_2916c9c5c01f26a00e722481ce70ef44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.27537572383880615]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.31011876463890076]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_8219f1b3815f08f4da10a4f0ae73e6c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.19937725365161896]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.14600974321365356]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_8641c44d1fdc5f5f6fde07a587682ddb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3637081980705261], [0.39640137553215027], [0.31833428144454956], [0.4412391483783722], [0.11331885308027267], [0.4164228141307831]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.269661545753479], [0.24576835334300995], [0.09308407455682755], [0.3879702687263489], [0.43500569462776184], [0.14424768090248108]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_286d30d54c383de0f3d6d38ef22979b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2915404736995697], [0.06591049581766129], [0.19010786712169647], [0.17498598992824554], [0.26003992557525635], [0.24949562549591064]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.25507786870002747], [0.1776416003704071], [0.0561361089348793], [0.35384827852249146], [0.1015116423368454], [0.23805175721645355]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_b9616e9bd6abd793fecaacac85e6c423(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28181594610214233], [0.256033331155777], [0.3946605324745178], [0.4794350862503052], [0.0947040542960167], [0.32221317291259766]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0779203251004219], [0.28568974137306213], [0.2259289026260376], [0.08137910068035126], [0.37925606966018677], [0.45194125175476074]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_a23e13b221b539b7eae7cf618fdd123a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4542856514453888], [0.32371410727500916], [0.47006481885910034], [0.24301113188266754], [0.31819093227386475], [0.3337242603302002]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4113561809062958], [0.38559842109680176], [0.34662380814552307], [0.25977855920791626], [0.09847179055213928], [0.15297983586788177]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_e7f3d7d11be6bef005875a691d131ce3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7f3d7d11be6bef005875a691d131ce3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb19a46cb4a00d0fdcc83e415ae800cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb19a46cb4a00d0fdcc83e415ae800cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_642a599041d29c06790004d0d1c62da8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_642a599041d29c06790004d0d1c62da8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_642a599041d29c06790004d0d1c62da8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_642a599041d29c06790004d0d1c62da8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53fa46e838ce4df94a0de0be6d560cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53fa46e838ce4df94a0de0be6d560cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce1679f204ee2ed40c0e27b987d8e637(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce1679f204ee2ed40c0e27b987d8e637(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce1679f204ee2ed40c0e27b987d8e637(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce1679f204ee2ed40c0e27b987d8e637(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e02e18c9d9fd67fcd494949797bcd6c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e02e18c9d9fd67fcd494949797bcd6c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e02e18c9d9fd67fcd494949797bcd6c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e02e18c9d9fd67fcd494949797bcd6c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04481b9bfaff5c8ada76d7111dc5b36d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04481b9bfaff5c8ada76d7111dc5b36d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbf43afc3c9144784bf2c425282bad97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbf43afc3c9144784bf2c425282bad97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17606de4874b9b0799f2069684810597(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25348156690597534], [0.1419389396905899], [0.21522848308086395], [0.18140600621700287], [0.35213175415992737]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1649070680141449], [0.39336472749710083], [0.47650161385536194], [0.22819465398788452], [0.16766460239887238]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_0a3de64a88b145cf53efc168b8c2c138(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.47471386194229126], [0.33389967679977417], [0.40402311086654663], [0.4485560953617096], [0.46584224700927734]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.21730045974254608], [0.35416045784950256], [0.32481035590171814], [0.011503858491778374], [0.4248878061771393]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_585419f76490241eaa26dc41626ed7c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18775731325149536], [0.12440189719200134], [0.05388212949037552], [0.37235766649246216], [0.08130541443824768]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.13528534770011902], [0.3804410994052887], [0.13431361317634583], [0.32307666540145874], [0.13232417404651642]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_a340aff25ead7c12a739411ae888407c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.21230792999267578], [0.0743480697274208], [0.0162972342222929], [0.3133012354373932], [0.06640253961086273]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.4135350286960602], [0.022249840199947357], [0.3079543709754944], [0.28616321086883545], [0.11313261836767197]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_2f9a60539947be8a12d4b323cde1640f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f9a60539947be8a12d4b323cde1640f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02bd2629126b35cdecb9b0fc7fea30da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02bd2629126b35cdecb9b0fc7fea30da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2a3a4d3dde8652ea816d52cbac75101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2a3a4d3dde8652ea816d52cbac75101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1482d847db1c0ceb2cbdb6e8f76529b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1482d847db1c0ceb2cbdb6e8f76529b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1482d847db1c0ceb2cbdb6e8f76529b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1482d847db1c0ceb2cbdb6e8f76529b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2203db1af31e047366ea29254c2aca06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2203db1af31e047366ea29254c2aca06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2203db1af31e047366ea29254c2aca06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2203db1af31e047366ea29254c2aca06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca968f17aaa054c6f237aa6e15e5df37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca968f17aaa054c6f237aa6e15e5df37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca968f17aaa054c6f237aa6e15e5df37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca968f17aaa054c6f237aa6e15e5df37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53fa46e838ce4df94a0de0be6d560cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53fa46e838ce4df94a0de0be6d560cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b66657045ea52b2a556d191be82db1e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.43446049094200134], [0.3114164471626282], [0.21863189339637756], [0.22740739583969116]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.05875740945339203], [0.3892061412334442], [0.20329052209854126], [0.3478715717792511]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_6658cd23850aaf776f1e34146854eeb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28952881693840027], [0.0999356284737587], [0.3750646412372589], [0.1684504747390747]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2514371871948242], [0.428699254989624], [0.0340176485478878], [0.16604164242744446]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f6812f4f9efeda0dc985d35fe13c6761(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1857021450996399], [0.232812762260437], [0.44483304023742676], [0.43192264437675476]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.37367141246795654], [0.26348745822906494], [0.22833964228630066], [0.3062015175819397]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_6eed9f9e94fbfee95574b0ecb9c87412(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28646060824394226], [0.2347613275051117], [0.07614689320325851], [0.21912746131420135]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.3861541450023651], [0.4490680992603302], [0.4151049554347992], [0.03166484087705612]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_7fcef654b21c0efbedb073d69fb47317(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fcef654b21c0efbedb073d69fb47317(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fcef654b21c0efbedb073d69fb47317(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fcef654b21c0efbedb073d69fb47317(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2a3a4d3dde8652ea816d52cbac75101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2a3a4d3dde8652ea816d52cbac75101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de0146d98124e8f73ad587a92ee850e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de0146d98124e8f73ad587a92ee850e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c31933cd677bc48ccc1f9ff263f5c915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c31933cd677bc48ccc1f9ff263f5c915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48438b32423b9b404aa368a29b369b68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48438b32423b9b404aa368a29b369b68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48438b32423b9b404aa368a29b369b68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48438b32423b9b404aa368a29b369b68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05bf30e0a86497370f67aa4f126e947c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c31933cd677bc48ccc1f9ff263f5c915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c31933cd677bc48ccc1f9ff263f5c915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bd0c7dece725587123f4604e106d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()