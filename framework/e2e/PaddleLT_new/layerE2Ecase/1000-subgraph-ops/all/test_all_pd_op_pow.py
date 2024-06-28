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
    class PrimitiveOp_1be2345881a25dfb1702d68739c68bc4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f16107909aefd3e56d39bedfa0b0d072(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1be2345881a25dfb1702d68739c68bc4
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2583e5ec1a4d6e23e8e4e30f644854f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1be2345881a25dfb1702d68739c68bc4
        def get_inputs(self):
            return [
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ff3499a1ff6763f7e2401273c1ad7097(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fb1802e122459f813b7b0533101b7466(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3499a1ff6763f7e2401273c1ad7097
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_36c89bbddac0b74b2eb9b10f3f56ddf2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 3)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2c1870d178d8f1823d3e8ce6a6e7b645(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36c89bbddac0b74b2eb9b10f3f56ddf2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.14134077727794647], [0.0687040314078331], [0.07436590641736984], [0.23809002339839935], [0.08811947703361511], [0.44216546416282654]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_fb1802e122459f813b7b0533101b7466(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3499a1ff6763f7e2401273c1ad7097
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_167ca4a63d1def51ab9d9543bd8110ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36c89bbddac0b74b2eb9b10f3f56ddf2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.04539785534143448], [0.1245993822813034], [0.31081730127334595], [0.2403320074081421], [0.3674183785915375], [0.13967715203762054]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_cc51a127de7939ca19c37928665f2fca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1be2345881a25dfb1702d68739c68bc4
        def get_inputs(self):
            return [
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_da4e9d49506e8ba5e2bb46f963bcbd01(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a7fd4e2f61291715e1d8ba908643dfa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da4e9d49506e8ba5e2bb46f963bcbd01
        def get_inputs(self):
            return [
                paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e4c3fbf0342a66452f764443a0334df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3499a1ff6763f7e2401273c1ad7097
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ded0f2823edc4942ece0db33156b8015(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da4e9d49506e8ba5e2bb46f963bcbd01
        def get_inputs(self):
            return [
                paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11b627a967509688265830179acb68c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1be2345881a25dfb1702d68739c68bc4
        def get_inputs(self):
            return [
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_549cf8ba833826ae32c017aaeea589f6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4374c07f26b8684b0fa7d0f2fb5c9e26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4374c07f26b8684b0fa7d0f2fb5c9e26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0c156217b3ef19a3102d529c6a0b462(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0c156217b3ef19a3102d529c6a0b462(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f36f95f24fa5df0f3935b94a12ec46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f36f95f24fa5df0f3935b94a12ec46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef54984c0c32450464b2a5632ccd6050(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef54984c0c32450464b2a5632ccd6050(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7122e989c9d7b03272686bcd2f1bfa11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7122e989c9d7b03272686bcd2f1bfa11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d5d8db17d71944d4d917137d0f35779(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d5d8db17d71944d4d917137d0f35779(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b676a42fdcc63defd002c38c2aec9d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b676a42fdcc63defd002c38c2aec9d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f941ace7cd955c306f8bcda0aa79ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f941ace7cd955c306f8bcda0aa79ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7122e989c9d7b03272686bcd2f1bfa11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7122e989c9d7b03272686bcd2f1bfa11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d5d8db17d71944d4d917137d0f35779(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d5d8db17d71944d4d917137d0f35779(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b676a42fdcc63defd002c38c2aec9d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b676a42fdcc63defd002c38c2aec9d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f941ace7cd955c306f8bcda0aa79ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f941ace7cd955c306f8bcda0aa79ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c585ed4daa6797df662a06c1ed82d48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1be2345881a25dfb1702d68739c68bc4
        def get_inputs(self):
            return [
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe5fd2b7d3400ec2bb13ca1e8af80bb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da4e9d49506e8ba5e2bb46f963bcbd01
        def get_inputs(self):
            return [
                paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5eda88a9fcb4157669483236434c2e21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1be2345881a25dfb1702d68739c68bc4
        def get_inputs(self):
            return [
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1690a28841068cdb3ac46f36bef041b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1be2345881a25dfb1702d68739c68bc4
        def get_inputs(self):
            return [
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21b027c2455c101144966365901f180c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1be2345881a25dfb1702d68739c68bc4
        def get_inputs(self):
            return [
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe657cf31e326f13c7024dc9467a8f1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1be2345881a25dfb1702d68739c68bc4
        def get_inputs(self):
            return [
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4374c07f26b8684b0fa7d0f2fb5c9e26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4374c07f26b8684b0fa7d0f2fb5c9e26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0c156217b3ef19a3102d529c6a0b462(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0c156217b3ef19a3102d529c6a0b462(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f36f95f24fa5df0f3935b94a12ec46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f36f95f24fa5df0f3935b94a12ec46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef54984c0c32450464b2a5632ccd6050(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef54984c0c32450464b2a5632ccd6050(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9331151fcd33a26f1a5458398aa50c6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da4e9d49506e8ba5e2bb46f963bcbd01
        def get_inputs(self):
            return [
                paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f16107909aefd3e56d39bedfa0b0d072(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1be2345881a25dfb1702d68739c68bc4
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c72e9a3672f98c1f5929f82b94500fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da4e9d49506e8ba5e2bb46f963bcbd01
        def get_inputs(self):
            return [
                paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dc530c336ff35c840a20700c34b7791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1be2345881a25dfb1702d68739c68bc4
        def get_inputs(self):
            return [
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ce34f5c82bc2da4ca33853f9182e2a7e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3800, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ef8158c7f919374dd9d7d32c0b32e9af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce34f5c82bc2da4ca33853f9182e2a7e
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9cd7122d4d2e17c4200891d3f3b0f60c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[150, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_67d05b01ea6e14fd8a0747257e1a8c5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cd7122d4d2e17c4200891d3f3b0f60c
        def get_inputs(self):
            return [
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d4e70bd237efb5747a22c1c8e0cdfb75(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 21824], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3941ddbf246ecc2e39d215acd0c8fb14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4e70bd237efb5747a22c1c8e0cdfb75
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5d35908fced7948e87121027a1d58ebe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 3)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_137cee31c616edbb366ce6464c34add6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d35908fced7948e87121027a1d58ebe
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.14134077727794647], [0.0687040314078331], [0.07436590641736984], [0.23809002339839935], [0.08811947703361511], [0.44216546416282654]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_3941ddbf246ecc2e39d215acd0c8fb14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4e70bd237efb5747a22c1c8e0cdfb75
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d830220855a919d6abfd61c225f38ee6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d35908fced7948e87121027a1d58ebe
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.04539785534143448], [0.1245993822813034], [0.31081730127334595], [0.2403320074081421], [0.3674183785915375], [0.13967715203762054]]], dtype='float32').reshape([1, 6, 1]),
            ]


    
    class PrimitiveOp_9710194255e51449f028c0266182e1ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1cded426ebe290176cd004bc72602f5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9710194255e51449f028c0266182e1ce
        def get_inputs(self):
            return [
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b4704aab5e4fa02d4f0bc45f1845ade4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3800, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_491c29e256d640c59787e64aef95b0f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4704aab5e4fa02d4f0bc45f1845ade4
        def get_inputs(self):
            return [
                paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e2eefb3ab92a6e04d7c80248a59b8d0a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 21824, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0bb7964a1febb85438ce4d96f8ddb30a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2eefb3ab92a6e04d7c80248a59b8d0a
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c59a8bbbf264786e1abfcc47bfcc11cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[15200, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f8ff0404b3dcf5c1b0daaaeff0ffeb78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c59a8bbbf264786e1abfcc47bfcc11cc
        def get_inputs(self):
            return [
                paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bf482e7aafe61656fa95204e9f41305a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[15200, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_798274d37a31db698582baef3ef90661(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf482e7aafe61656fa95204e9f41305a
        def get_inputs(self):
            return [
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_176be4e85d458250ae5cd99d2fa4e500(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 64, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f098c2e51518d7c0adc9f08bf436f165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_176be4e85d458250ae5cd99d2fa4e500
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f098c2e51518d7c0adc9f08bf436f165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_176be4e85d458250ae5cd99d2fa4e500
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_780b1f0b0ef9248df5d507bdca6d50f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 128, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_706485a87cbf8b82f8f8f682a73dda27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_780b1f0b0ef9248df5d507bdca6d50f0
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_706485a87cbf8b82f8f8f682a73dda27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_780b1f0b0ef9248df5d507bdca6d50f0
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4d491a4a8a1f9954ea56a8bbd3731802(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 256, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c8907aa6576206c4991cfb78c72c215f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d491a4a8a1f9954ea56a8bbd3731802
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8907aa6576206c4991cfb78c72c215f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d491a4a8a1f9954ea56a8bbd3731802
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0a83bffeeb177dc0897b1c8e0c02cffa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 512, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_773308390580d04b8c0bc03c7d463e8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a83bffeeb177dc0897b1c8e0c02cffa
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_773308390580d04b8c0bc03c7d463e8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a83bffeeb177dc0897b1c8e0c02cffa
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_183576c20da80829a850d5c75f38772a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 64, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e0cbe4379dbf3436b5732d927cf2f8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_183576c20da80829a850d5c75f38772a
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e0cbe4379dbf3436b5732d927cf2f8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_183576c20da80829a850d5c75f38772a
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c7ee3f27d49b52733dcebe5325fe3b3d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 128, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9409e71b3eda3d19adc5f8fe63fe665(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ee3f27d49b52733dcebe5325fe3b3d
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9409e71b3eda3d19adc5f8fe63fe665(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ee3f27d49b52733dcebe5325fe3b3d
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9b8ef5cb938761734dbf6e374d58d617(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 256, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d5746781b789e328c59b9f0a36358186(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b8ef5cb938761734dbf6e374d58d617
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5746781b789e328c59b9f0a36358186(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b8ef5cb938761734dbf6e374d58d617
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_270f5ae774e50580b502304edc4e21af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 512, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_03645c1855f7e6453df2db111ecf1bf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_270f5ae774e50580b502304edc4e21af
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03645c1855f7e6453df2db111ecf1bf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_270f5ae774e50580b502304edc4e21af
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e0cbe4379dbf3436b5732d927cf2f8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_183576c20da80829a850d5c75f38772a
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e0cbe4379dbf3436b5732d927cf2f8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_183576c20da80829a850d5c75f38772a
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9409e71b3eda3d19adc5f8fe63fe665(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ee3f27d49b52733dcebe5325fe3b3d
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9409e71b3eda3d19adc5f8fe63fe665(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ee3f27d49b52733dcebe5325fe3b3d
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5746781b789e328c59b9f0a36358186(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b8ef5cb938761734dbf6e374d58d617
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5746781b789e328c59b9f0a36358186(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b8ef5cb938761734dbf6e374d58d617
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03645c1855f7e6453df2db111ecf1bf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_270f5ae774e50580b502304edc4e21af
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03645c1855f7e6453df2db111ecf1bf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_270f5ae774e50580b502304edc4e21af
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dc7ef3bbacdfb10f35134f46b8a013cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2204, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_67b736f2839ec4b90ce5157d2505398a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc7ef3bbacdfb10f35134f46b8a013cd
        def get_inputs(self):
            return [
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_97d7df0ca49ff896851738ef21cdadb0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[70, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bfcee09ee8c072913efc1dc950144bf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97d7df0ca49ff896851738ef21cdadb0
        def get_inputs(self):
            return [
                paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e58036e7173fc1825a63a2b75e3a34ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[551, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a880aedebe513b16aa877cc6001320bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e58036e7173fc1825a63a2b75e3a34ee
        def get_inputs(self):
            return [
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c249096a247ea34a823bb0ebfc1b98ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[247, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c407127a5574affee786d7d595b60c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c249096a247ea34a823bb0ebfc1b98ec
        def get_inputs(self):
            return [
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d4c62ab6129c98b1a0b7518b0b7b6c8c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[950, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8a245009a37155aa00c70977ead687ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4c62ab6129c98b1a0b7518b0b7b6c8c
        def get_inputs(self):
            return [
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_98d2789c0be829619ae3f3c3991e0faf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8816, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a5f02d810708a306f627ae27af100443(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d2789c0be829619ae3f3c3991e0faf
        def get_inputs(self):
            return [
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f098c2e51518d7c0adc9f08bf436f165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_176be4e85d458250ae5cd99d2fa4e500
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f098c2e51518d7c0adc9f08bf436f165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_176be4e85d458250ae5cd99d2fa4e500
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_706485a87cbf8b82f8f8f682a73dda27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_780b1f0b0ef9248df5d507bdca6d50f0
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_706485a87cbf8b82f8f8f682a73dda27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_780b1f0b0ef9248df5d507bdca6d50f0
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8907aa6576206c4991cfb78c72c215f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d491a4a8a1f9954ea56a8bbd3731802
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8907aa6576206c4991cfb78c72c215f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d491a4a8a1f9954ea56a8bbd3731802
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_773308390580d04b8c0bc03c7d463e8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a83bffeeb177dc0897b1c8e0c02cffa
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_773308390580d04b8c0bc03c7d463e8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a83bffeeb177dc0897b1c8e0c02cffa
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e76a5fadcc1bb93c4348a62a9034379b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[247, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cf1dbe17b7da46cd54a513518e14fd9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e76a5fadcc1bb93c4348a62a9034379b
        def get_inputs(self):
            return [
                paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef8158c7f919374dd9d7d32c0b32e9af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce34f5c82bc2da4ca33853f9182e2a7e
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fec8f6e8796a96598737b3a60302640c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[950, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_71a0574afd1bcb4112cae29ae60504d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fec8f6e8796a96598737b3a60302640c
        def get_inputs(self):
            return [
                paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9eb6b575ebd2b2d69839891c07830ca4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[70, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f32c2c1b04ad67f80c99d7f9499c148(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9eb6b575ebd2b2d69839891c07830ca4
        def get_inputs(self):
            return [
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8040c3bfbb500c986fda87f1988d9936(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9b52484fcf10ee4d3b41eade8cce3d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8040c3bfbb500c986fda87f1988d9936
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04f92848f6fb118b251ffef2a2934930(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8040c3bfbb500c986fda87f1988d9936
        def get_inputs(self):
            return [
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb1802e122459f813b7b0533101b7466(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3499a1ff6763f7e2401273c1ad7097
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c1870d178d8f1823d3e8ce6a6e7b645(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36c89bbddac0b74b2eb9b10f3f56ddf2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.14134077727794647], [0.0687040314078331], [0.07436590641736984], [0.23809002339839935], [0.08811947703361511], [0.44216546416282654]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_fb1802e122459f813b7b0533101b7466(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3499a1ff6763f7e2401273c1ad7097
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_167ca4a63d1def51ab9d9543bd8110ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36c89bbddac0b74b2eb9b10f3f56ddf2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.04539785534143448], [0.1245993822813034], [0.31081730127334595], [0.2403320074081421], [0.3674183785915375], [0.13967715203762054]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_0779b22d82e5228ed95382a62002c2d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8040c3bfbb500c986fda87f1988d9936
        def get_inputs(self):
            return [
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6dd99f890689320b8041fdbf29d84d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8040c3bfbb500c986fda87f1988d9936
        def get_inputs(self):
            return [
                paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e4c3fbf0342a66452f764443a0334df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3499a1ff6763f7e2401273c1ad7097
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b220fdc9d9ebf921147eb3e7acb9a1ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8040c3bfbb500c986fda87f1988d9936
        def get_inputs(self):
            return [
                paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fe02bb873c74bab894dc847eeaaacc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8040c3bfbb500c986fda87f1988d9936
        def get_inputs(self):
            return [
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4374c07f26b8684b0fa7d0f2fb5c9e26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4374c07f26b8684b0fa7d0f2fb5c9e26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0c156217b3ef19a3102d529c6a0b462(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0c156217b3ef19a3102d529c6a0b462(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f36f95f24fa5df0f3935b94a12ec46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f36f95f24fa5df0f3935b94a12ec46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef54984c0c32450464b2a5632ccd6050(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef54984c0c32450464b2a5632ccd6050(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7122e989c9d7b03272686bcd2f1bfa11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7122e989c9d7b03272686bcd2f1bfa11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d5d8db17d71944d4d917137d0f35779(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d5d8db17d71944d4d917137d0f35779(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b676a42fdcc63defd002c38c2aec9d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b676a42fdcc63defd002c38c2aec9d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f941ace7cd955c306f8bcda0aa79ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f941ace7cd955c306f8bcda0aa79ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7122e989c9d7b03272686bcd2f1bfa11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7122e989c9d7b03272686bcd2f1bfa11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d5d8db17d71944d4d917137d0f35779(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d5d8db17d71944d4d917137d0f35779(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b676a42fdcc63defd002c38c2aec9d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b676a42fdcc63defd002c38c2aec9d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f941ace7cd955c306f8bcda0aa79ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f941ace7cd955c306f8bcda0aa79ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c54526db94244e19eb7f63960e4b2554(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8040c3bfbb500c986fda87f1988d9936
        def get_inputs(self):
            return [
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d07fd514ad9accc6ad0015301000c26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8040c3bfbb500c986fda87f1988d9936
        def get_inputs(self):
            return [
                paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2c88e92cdd52c8d4e4452be4c03ff0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8040c3bfbb500c986fda87f1988d9936
        def get_inputs(self):
            return [
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_253384bc823ef3d4dc938e013ed4213f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8040c3bfbb500c986fda87f1988d9936
        def get_inputs(self):
            return [
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e1bc5567a1c4e6253fe45a0b86c3137(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8040c3bfbb500c986fda87f1988d9936
        def get_inputs(self):
            return [
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_135cd090148d527c49dd455cbc50c0eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8040c3bfbb500c986fda87f1988d9936
        def get_inputs(self):
            return [
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4374c07f26b8684b0fa7d0f2fb5c9e26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4374c07f26b8684b0fa7d0f2fb5c9e26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0c156217b3ef19a3102d529c6a0b462(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0c156217b3ef19a3102d529c6a0b462(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f36f95f24fa5df0f3935b94a12ec46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f36f95f24fa5df0f3935b94a12ec46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef54984c0c32450464b2a5632ccd6050(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef54984c0c32450464b2a5632ccd6050(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549cf8ba833826ae32c017aaeea589f6
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_968bc1a16f69363865434978850e4363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8040c3bfbb500c986fda87f1988d9936
        def get_inputs(self):
            return [
                paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b52484fcf10ee4d3b41eade8cce3d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8040c3bfbb500c986fda87f1988d9936
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1827970bb780b0ea48195faf54049535(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8040c3bfbb500c986fda87f1988d9936
        def get_inputs(self):
            return [
                paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6553ab196c28c1c4d0346c1ece61ab4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8040c3bfbb500c986fda87f1988d9936
        def get_inputs(self):
            return [
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()