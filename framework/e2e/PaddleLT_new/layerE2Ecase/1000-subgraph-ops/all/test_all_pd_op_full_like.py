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
    class PrimitiveOp_c06a42c8d7007fe5b2423ce001701566(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 20.0
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b585acc8d2350f7653050f46b932ffcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c06a42c8d7007fe5b2423ce001701566
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            ]


    
    class PrimitiveOp_01ce3850f5c55d34e98e82895c50ec7b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4a8838f40f912cb0de048c4d135a8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01ce3850f5c55d34e98e82895c50ec7b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            ]


    class TestPrimitiveOp_4a8838f40f912cb0de048c4d135a8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01ce3850f5c55d34e98e82895c50ec7b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            ]


    
    class PrimitiveOp_821ae24c866028664d2d8a59ce4a4879(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            return paddle._C_ops.full_like(input_0, input_1, paddle.bool, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_297e17301032126b9dff52a9cb868b4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_821ae24c866028664d2d8a59ce4a4879
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_bdf6bda0cd20e5a75594826d782c5867(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            return paddle._C_ops.full_like(input_0, input_1, paddle.float32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0796256d643eb66cb84166a1418f16ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bdf6bda0cd20e5a75594826d782c5867
        def get_inputs(self):
            return [
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fe346a01089c32c408d32912f74bc448(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            ]


    class TestPrimitiveOp_fe346a01089c32c408d32912f74bc448(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            ]


    class TestPrimitiveOp_21f70ab31edc37748bec5378fc4c6035(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            ]


    class TestPrimitiveOp_21f70ab31edc37748bec5378fc4c6035(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            ]


    class TestPrimitiveOp_cadca359b48d21f74e3fc5c55270d9ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            ]


    class TestPrimitiveOp_cadca359b48d21f74e3fc5c55270d9ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            ]


    
    class PrimitiveOp_f5061e5d26855eb10e191f0e3d62f063(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 80.0
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_616b395c3a1ade1914dbb02d6e4ed8e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5061e5d26855eb10e191f0e3d62f063
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    
    class PrimitiveOp_969fcd64d8e12070b7e6fd47e044228c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5e7c77624c742188ea04c1bc5654970a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_969fcd64d8e12070b7e6fd47e044228c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    class TestPrimitiveOp_5e7c77624c742188ea04c1bc5654970a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_969fcd64d8e12070b7e6fd47e044228c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    class TestPrimitiveOp_8abd9aa4473ce22b4ee14cdf73c730f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_821ae24c866028664d2d8a59ce4a4879
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_7f955e00fbb0bee73cb43f32d9ac71ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 20.0
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_59d4cd236702b7cc767d737d71369f25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f955e00fbb0bee73cb43f32d9ac71ec
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    
    class PrimitiveOp_a0303a8a71b128d50b8467aea244b5e0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c0bb1ce1d2e998e82d456455d3ca0da6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0303a8a71b128d50b8467aea244b5e0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    class TestPrimitiveOp_c0bb1ce1d2e998e82d456455d3ca0da6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0303a8a71b128d50b8467aea244b5e0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    class TestPrimitiveOp_37c8cf2117a8139f8a87fef295e4fc7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_821ae24c866028664d2d8a59ce4a4879
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b9aca0827ade29f38eedf47a1d60548f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            ]


    class TestPrimitiveOp_b9aca0827ade29f38eedf47a1d60548f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            ]


    
    class PrimitiveOp_307b8c7b23508b8f4a98761dcaaf2288(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 20.0
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_38facb067fed7b8d37fe8384ae02e391(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_307b8c7b23508b8f4a98761dcaaf2288
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            ]


    
    class PrimitiveOp_66c91ec58b1457e7a27c9cc29245cd26(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7489b833f67f709d4e76b3b97da8e0ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66c91ec58b1457e7a27c9cc29245cd26
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            ]


    class TestPrimitiveOp_7489b833f67f709d4e76b3b97da8e0ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66c91ec58b1457e7a27c9cc29245cd26
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            ]


    class TestPrimitiveOp_297e17301032126b9dff52a9cb868b4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_821ae24c866028664d2d8a59ce4a4879
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_f246e2618e47763cdfcc6b3b5df89159(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            return paddle._C_ops.full_like(input_0, input_1, paddle.float32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_539e08d90745c2d1eaa7d4ce7223ad87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f246e2618e47763cdfcc6b3b5df89159
        def get_inputs(self):
            return [
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe346a01089c32c408d32912f74bc448(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            ]


    class TestPrimitiveOp_fe346a01089c32c408d32912f74bc448(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            ]


    class TestPrimitiveOp_21f70ab31edc37748bec5378fc4c6035(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            ]


    class TestPrimitiveOp_21f70ab31edc37748bec5378fc4c6035(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            ]


    class TestPrimitiveOp_cadca359b48d21f74e3fc5c55270d9ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            ]


    class TestPrimitiveOp_cadca359b48d21f74e3fc5c55270d9ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            ]


    
    class PrimitiveOp_beeb722b72e33ca00555ab698496afff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 80.0
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb604100d0a7bdf75fdd38725539df13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_beeb722b72e33ca00555ab698496afff
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    class TestPrimitiveOp_f71ca2790e100b0c25d1009fe9570bc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66c91ec58b1457e7a27c9cc29245cd26
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    class TestPrimitiveOp_f71ca2790e100b0c25d1009fe9570bc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66c91ec58b1457e7a27c9cc29245cd26
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    class TestPrimitiveOp_8abd9aa4473ce22b4ee14cdf73c730f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_821ae24c866028664d2d8a59ce4a4879
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d89e6d316355812af4b680bbf9c45a70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_307b8c7b23508b8f4a98761dcaaf2288
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    class TestPrimitiveOp_4bbf00d58c5d9313621ef45d1e97dfd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66c91ec58b1457e7a27c9cc29245cd26
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    class TestPrimitiveOp_4bbf00d58c5d9313621ef45d1e97dfd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66c91ec58b1457e7a27c9cc29245cd26
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    class TestPrimitiveOp_37c8cf2117a8139f8a87fef295e4fc7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_821ae24c866028664d2d8a59ce4a4879
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b9aca0827ade29f38eedf47a1d60548f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            ]


    class TestPrimitiveOp_b9aca0827ade29f38eedf47a1d60548f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665e3b17c44fd3f686ecec3249f117cb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            ]


    

if __name__ == '__main__':
    unittest.main()