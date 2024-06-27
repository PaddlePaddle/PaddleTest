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
    class PrimitiveOp_239cb8a32eafd33e60b806689ac66bf7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_48e53ddfe8e6f71bbf22b6cdeb89dff6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_239cb8a32eafd33e60b806689ac66bf7
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_4054e272cbcc9a98a5004f1ded77e130(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_239cb8a32eafd33e60b806689ac66bf7
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4b872add75a329747f6290c663416b69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_a4a66d5bae5a264680d65e8536ac291f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[150], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9094c3135bc3380270383eaac3ae6ba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3e63c7cdebe34dbaa662fb943b640c67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d40c9bbeaa848ae8e9ce9ef717b0a4c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f0f88ff64322f70fd3c8de98d7c2f5f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_00561a230a904c819ce5a6bdebd08de7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_239cb8a32eafd33e60b806689ac66bf7
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_c0c40d13075bb8c55c089ac60f3d2f73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2204], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_efeec48d7857f09889f9df71266601cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_bb905f1d3cf659c2f77a34980aaa6c1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[551], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c4c4a8b8628f3735914dc72812cbad01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_60999340c16eef39dae424d1ca02422e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_2bd9f2e603497484f408c30690a665cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[8816], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_a7816b89df4e13a1ce02cd99b8addda4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_239cb8a32eafd33e60b806689ac66bf7
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_166fdbf6e602ce4b731ce4958b8a39a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_239cb8a32eafd33e60b806689ac66bf7
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_f8fab262bb2da351a9961a5bdbb643e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_4b872add75a329747f6290c663416b69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_69decf4cde6698fd39813aeb9e5479c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_56c1eca3cfc2000e46fcc67a6f331028(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_dcbccff4b525ffe1601c3a5e6f18d320(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_73e33208b95d702107206a1aca43921c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcbccff4b525ffe1601c3a5e6f18d320
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_be380fb7cb676ed83abf1cf11e73be46(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4096, 5], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bdcc68d313c0c82741bb04ffefd7ac90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be380fb7cb676ed83abf1cf11e73be46
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_f1c89686c231a42c9c6246b45f8f9bcb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3800], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1676b5040af9027720d3bc36cb829133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1c89686c231a42c9c6246b45f8f9bcb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_609645b7b1c4f675d650eb3997d5c65f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[150], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd63114ad3d3132d438023f5d4d6e2e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_609645b7b1c4f675d650eb3997d5c65f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[150], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_1eb593d31a04a87d736d2c3457b31cec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ec9661cd1d3674f61ad378711087b89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1eb593d31a04a87d736d2c3457b31cec
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c85ad1529a406391c99033e934aa5b8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1c89686c231a42c9c6246b45f8f9bcb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_65ed7c8a3d5bbad5930589b73bdfd984(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[15200], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb6ba338fe910fd8404ac88c45ea6875(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65ed7c8a3d5bbad5930589b73bdfd984
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7a9ff957846f02f01ea35fbaa1c3e51f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65ed7c8a3d5bbad5930589b73bdfd984
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_1a6e44a311ac6a32b2c14b5453674e2e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[64, 5], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_049f3143d0f36528d5a258ef8eb1b1e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a6e44a311ac6a32b2c14b5453674e2e
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_03382a90ea2d4f0857a2db42d320b025(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2204], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_57babe3f1437ad3620ce1881dc01d5a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03382a90ea2d4f0857a2db42d320b025
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2204], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_c7c950a92b5b19a747de113072e1ff21(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[70], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ec731a195187479bf22d4be6a9acc3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7c950a92b5b19a747de113072e1ff21
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_d4063d7e30180684f9623406db349a2d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[551], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ca6a3eb1812d7b999dd84bd6715a366(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4063d7e30180684f9623406db349a2d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[551], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_ee3cdc1bbf6f685ad4b3c9706c801fc8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[247], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0ebabc297b16d6e65dd18bad9a68b1ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee3cdc1bbf6f685ad4b3c9706c801fc8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_a34a6b92166ef7cfbe0dedffef35cdb6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[950], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_645197def6e261966a19c7d73311c7b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a34a6b92166ef7cfbe0dedffef35cdb6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_b9f051ba132aff146ef2ec42e1995c5e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8816], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce030dcb6adc20c0c3d456c5b99e3791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9f051ba132aff146ef2ec42e1995c5e
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[8816], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_1001da4ca46a5ccd423ec2ba53cdf6d0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16384, 5], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b200a980eec65a55fe709a55e3667de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1001da4ca46a5ccd423ec2ba53cdf6d0
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_73eb21ba587052399aa8d79b435e7e98(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[256, 5], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_718727add42a1610a9799aad95a2dcfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73eb21ba587052399aa8d79b435e7e98
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_702c22458aca2261de2ebbf3e3897be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee3cdc1bbf6f685ad4b3c9706c801fc8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_1676b5040af9027720d3bc36cb829133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1c89686c231a42c9c6246b45f8f9bcb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_bb60178c5351f4b8ddc7e2f1a3f6a5b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a34a6b92166ef7cfbe0dedffef35cdb6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_03edcab50b9d1e289dfa641a8cc15d26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7c950a92b5b19a747de113072e1ff21
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_48e53ddfe8e6f71bbf22b6cdeb89dff6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_239cb8a32eafd33e60b806689ac66bf7
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_4054e272cbcc9a98a5004f1ded77e130(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_239cb8a32eafd33e60b806689ac66bf7
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_4b872add75a329747f6290c663416b69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_a4a66d5bae5a264680d65e8536ac291f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[150], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9094c3135bc3380270383eaac3ae6ba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3e63c7cdebe34dbaa662fb943b640c67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d40c9bbeaa848ae8e9ce9ef717b0a4c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f0f88ff64322f70fd3c8de98d7c2f5f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_00561a230a904c819ce5a6bdebd08de7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_239cb8a32eafd33e60b806689ac66bf7
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_c0c40d13075bb8c55c089ac60f3d2f73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2204], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_efeec48d7857f09889f9df71266601cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_bb905f1d3cf659c2f77a34980aaa6c1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[551], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c4c4a8b8628f3735914dc72812cbad01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_60999340c16eef39dae424d1ca02422e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_2bd9f2e603497484f408c30690a665cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[8816], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_a7816b89df4e13a1ce02cd99b8addda4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_239cb8a32eafd33e60b806689ac66bf7
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_166fdbf6e602ce4b731ce4958b8a39a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_239cb8a32eafd33e60b806689ac66bf7
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_f8fab262bb2da351a9961a5bdbb643e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_4b872add75a329747f6290c663416b69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_69decf4cde6698fd39813aeb9e5479c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_56c1eca3cfc2000e46fcc67a6f331028(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e757f12226b4e2e4a14d4782171fa78
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    

if __name__ == '__main__':
    unittest.main()