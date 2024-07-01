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
    class PrimitiveOp_39335ad63479fde5528387573ffd4081(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle.arange(input_0, input_1, input_2, dtype=paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e51de6bba81cbe88a6f81456a57fcc84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e51de6bba81cbe88a6f81456a57fcc84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3e126469ba5ceda95f98dfd0a962db31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3e126469ba5ceda95f98dfd0a962db31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e527d8eb5e61c9a9a8abe284c52c0c6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e527d8eb5e61c9a9a8abe284c52c0c6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_d7251295a466faa5c273b45c5fad3af2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle.arange(input_0, input_1, input_2, dtype=paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8614ba89d7853189423f2fe4353f73dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_79a2efe5fd56a079854c57c1344f20b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([96.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_52eaf7e7dc2f393f81e957d0951f8713(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([48.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0cdcdfe8ef6f210255ced71957582839(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([24.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e3737a7f458f2fe73e66ed304a72a828(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([64.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a25548491ad7884f05517671ccf43a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([32.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1a08667604a9db8705effd7e9d0834ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([16.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f8ee944247753a423d062405ce31a9b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([80.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_239b190385f51fc9b73286adc47ab936(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([40.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8614ba89d7853189423f2fe4353f73dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e51de6bba81cbe88a6f81456a57fcc84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e51de6bba81cbe88a6f81456a57fcc84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3e126469ba5ceda95f98dfd0a962db31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3e126469ba5ceda95f98dfd0a962db31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e527d8eb5e61c9a9a8abe284c52c0c6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e527d8eb5e61c9a9a8abe284c52c0c6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f8ee944247753a423d062405ce31a9b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([80.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_83b22cbc6d6f1a43aaa7e0fbb21d3f87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([14.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b1cdb809628e0c9a54b985a2857ec567(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([28.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7b657b786577e426057594124636b6d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([56.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8614ba89d7853189423f2fe4353f73dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_c8867d9065ea967d34811d0ab490f51d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle.arange(input_0, input_1, input_2, dtype=paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_917e07011bdb00de4130522ec2f366ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8867d9065ea967d34811d0ab490f51d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([24.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a9fbd7b053c0455f339a1c2f5f24ec1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([68.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_52a7ab96a395e27578076a2cc8a4f4d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([34.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7d52e296906807330d310d8baef92f55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([17.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e527d8eb5e61c9a9a8abe284c52c0c6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e527d8eb5e61c9a9a8abe284c52c0c6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3e126469ba5ceda95f98dfd0a962db31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3e126469ba5ceda95f98dfd0a962db31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e51de6bba81cbe88a6f81456a57fcc84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e51de6bba81cbe88a6f81456a57fcc84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0ad293af35a282a48b41b539c02d4692(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0ad293af35a282a48b41b539c02d4692(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d8743ee2b6c8cc2f4a1b86afa9bb4e38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d8743ee2b6c8cc2f4a1b86afa9bb4e38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_7a8336dd4c792257232376f268e1f579(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([152.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_18a95221fd500ffc939c712ad028f29f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([100.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a4abafc7b82308506431d5e0c52a5a6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([76.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d497b5d9e470fbeb5551a80b6cd93ac0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([50.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_098fd066f8124fbca2682f83e57ea9ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([38.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7684b661195b1bf8d2c05b235fefff42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([25.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c2c938e791efe60c8938bbc8638445ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([19.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b56ee5526ff384c66f36659ef30fcaa5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([13.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_07774d1b7125221fa2e1fb026921657d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([10.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8c2f28592c781403630228cce0dc9a9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e51de6bba81cbe88a6f81456a57fcc84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e51de6bba81cbe88a6f81456a57fcc84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3e126469ba5ceda95f98dfd0a962db31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3e126469ba5ceda95f98dfd0a962db31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e527d8eb5e61c9a9a8abe284c52c0c6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e527d8eb5e61c9a9a8abe284c52c0c6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d7645bf373fdaa208a3cbffe1c82c26d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([72.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a8381f8aed6345188e686cfffc42107a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([36.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1120069fb461667cee5eb4bb9f7bbcb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7251295a466faa5c273b45c5fad3af2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([18.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e51de6bba81cbe88a6f81456a57fcc84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e51de6bba81cbe88a6f81456a57fcc84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3e126469ba5ceda95f98dfd0a962db31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3e126469ba5ceda95f98dfd0a962db31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e527d8eb5e61c9a9a8abe284c52c0c6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e527d8eb5e61c9a9a8abe284c52c0c6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39335ad63479fde5528387573ffd4081
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_f1519458718bce828ba4d734a268c4f7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle.arange(input_0, input_1, input_2, dtype=paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_710c058c8fc3d9c2e2a9a5a95f301f59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_710c058c8fc3d9c2e2a9a5a95f301f59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_16f3e4d34b18ae218f8393e1d8f1c70e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_16f3e4d34b18ae218f8393e1d8f1c70e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_47e328002215f61e61a18000dcc6d5e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_47e328002215f61e61a18000dcc6d5e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle.arange(input_0, input_1, input_2, dtype=paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fb29aa1ec67aed99fb0acd3481c2bf43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f52955f21f109cf50a0a3cbbf58fed12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([96.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_18d5ecc0419b37937893d9321d10055c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([48.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d9be321699bbfb77aad71ae37649ca50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([24.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3c123022f124ab69170d78e4008d3319(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([64.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fccd317468024c13931b03ba410b8832(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([32.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bbbe871d5a00bc521fef9e8850c893fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([16.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0c79c167fbcf06f88619043942d2c421(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([80.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d9476744dcf75232de07d6afd0983c65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([40.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fb29aa1ec67aed99fb0acd3481c2bf43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_710c058c8fc3d9c2e2a9a5a95f301f59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_710c058c8fc3d9c2e2a9a5a95f301f59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_16f3e4d34b18ae218f8393e1d8f1c70e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_16f3e4d34b18ae218f8393e1d8f1c70e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_47e328002215f61e61a18000dcc6d5e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_47e328002215f61e61a18000dcc6d5e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0c79c167fbcf06f88619043942d2c421(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([80.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f9aee4907474ac5610a549f2061fc3bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([14.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9c46bd403fb2f0c0a6e3c8e50fb68df2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([28.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_72308e0fbd3fa83608cc5a5dc46fc3d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([56.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fb29aa1ec67aed99fb0acd3481c2bf43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_f39097f04d3c3815aa534ecedb5c7394(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle.arange(input_0, input_1, input_2, dtype=paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d72845cf919f4de56b460e32813f6964(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f39097f04d3c3815aa534ecedb5c7394
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([24.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_63957a0c219dd59ad77a67e83b86fd32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([68.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_127c39929a7c0fa3657f74340f033d4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([34.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_af4afbc1fd97e4b514f48df49fbb5d98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([17.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_47e328002215f61e61a18000dcc6d5e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_47e328002215f61e61a18000dcc6d5e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_16f3e4d34b18ae218f8393e1d8f1c70e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_16f3e4d34b18ae218f8393e1d8f1c70e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_710c058c8fc3d9c2e2a9a5a95f301f59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_710c058c8fc3d9c2e2a9a5a95f301f59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f2f7a0c20260b5d298cbc7e30f783e5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f2f7a0c20260b5d298cbc7e30f783e5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_23bb9c3f81ded01fa00f9d442f098f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_23bb9c3f81ded01fa00f9d442f098f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5f2e03e7e487146fd27bb9796661b57e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([152.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1c1332f825ad23c84d8690d5615bae77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([100.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_142b70c0d5968a5936d7b667d605f7cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([76.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ffed11ba74d2d066d08c6fd09c281b47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([50.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7b951e06d719cecf2a774a03807664ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([38.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_aa4f3feaad4c66c0d99491b6eb5b3464(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([25.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fab249d460a49d2e53c011d9765e29d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([19.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_511bc282d11b13d580a55fc277f5cb62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([13.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d41d3daccec9d0f36e7299fe3c2050fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([10.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6e1fc5cce7bed03276717a923136970a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_710c058c8fc3d9c2e2a9a5a95f301f59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_710c058c8fc3d9c2e2a9a5a95f301f59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_16f3e4d34b18ae218f8393e1d8f1c70e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_16f3e4d34b18ae218f8393e1d8f1c70e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_47e328002215f61e61a18000dcc6d5e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_47e328002215f61e61a18000dcc6d5e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_051e0ff03fd36a88fa6dde4d0099f76a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([72.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_62734fbefd63ddbffb11ec747cf17ff7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([36.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1983da376b26d08beb26d3ad56e36b9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fb672b6fc740dd4bad92c5b16f6dfa7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([18.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_710c058c8fc3d9c2e2a9a5a95f301f59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_710c058c8fc3d9c2e2a9a5a95f301f59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_16f3e4d34b18ae218f8393e1d8f1c70e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_16f3e4d34b18ae218f8393e1d8f1c70e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_47e328002215f61e61a18000dcc6d5e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_47e328002215f61e61a18000dcc6d5e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1519458718bce828ba4d734a268c4f7
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()