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
    class PrimitiveOp_7f238ec24b96889b3b4769516cbdefe2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8452f64623aefffa8625e6f7487a6853(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f238ec24b96889b3b4769516cbdefe2
        def get_inputs(self):
            return [
                paddle.uniform([4, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_19da5067fb03899b27ffeaf39ee13c46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7589a322ee1911aca73c2449297a0319(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[150], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d7be639a2f39c26098cbb9d5f2247fda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[86970], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d7be639a2f39c26098cbb9d5f2247fda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[86970], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b00cebb4afa1be06185c7c244065d210(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[242991], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b00cebb4afa1be06185c7c244065d210(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[242991], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_12658cf1c3e59813bcdfab500ea9e843(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[40], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_19da5067fb03899b27ffeaf39ee13c46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_86c8dc2666f0a25531ced25690d379d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[220968], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_86c8dc2666f0a25531ced25690d379d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[220968], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_aed3d7d935664c666fd48fb955c751c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[153450], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_aed3d7d935664c666fd48fb955c751c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[153450], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_53ba4dd7d52fdacaf70b75de9345b1b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f238ec24b96889b3b4769516cbdefe2
        def get_inputs(self):
            return [
                paddle.uniform([3, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f5c25dcc0198f2ebb21d88f8991c9b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185691], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_2f5c25dcc0198f2ebb21d88f8991c9b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185691], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_25bd60a02307bb0e709db6d34b6b3d87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[113061], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_25bd60a02307bb0e709db6d34b6b3d87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[113061], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6da79125a5125270b49eecf31e3b3a0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6da79125a5125270b49eecf31e3b3a0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_275363c30eb80930f6ab19437e29c649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[205923], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_275363c30eb80930f6ab19437e29c649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[205923], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_725d0f28087b3ae525fac4fed3e39050(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[2204], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9fb25636c6b0d65806f236343802dc5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[123783], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9fb25636c6b0d65806f236343802dc5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[123783], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a353172979bf388fd2db7e98d749830d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[171888], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a353172979bf388fd2db7e98d749830d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[171888], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ea07b997cb13dc057aa05ee450ba2ec6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_425540fdeb361e1c9bc1a39e243ed555(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[551], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c37dbc2640398d098a134dbefaa024cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[217413], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c37dbc2640398d098a134dbefaa024cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[217413], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d95f601cc73b8a54e9c0c1b91bb66ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7840b8524711b7381dea3d5ddc25b8ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_bc3a79c58f61b3af180cec0fabc28819(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[8816], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_63cd670cc743c6b4b345cd207bf587cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f238ec24b96889b3b4769516cbdefe2
        def get_inputs(self):
            return [
                paddle.uniform([6, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5eea8927b783e3f7b8256accf70336ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f238ec24b96889b3b4769516cbdefe2
        def get_inputs(self):
            return [
                paddle.uniform([2, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d95f601cc73b8a54e9c0c1b91bb66ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_19da5067fb03899b27ffeaf39ee13c46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7840b8524711b7381dea3d5ddc25b8ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ea07b997cb13dc057aa05ee450ba2ec6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0488837e9206f26b5e76c56ecb5965cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185658], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0488837e9206f26b5e76c56ecb5965cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185658], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_3ca6daa6b6602ca1b52d093d1e0c07c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d626f07dc028c2e7e1235391981fd02e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ca6daa6b6602ca1b52d093d1e0c07c7
        def get_inputs(self):
            return [
                paddle.uniform([4, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19da5067fb03899b27ffeaf39ee13c46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7589a322ee1911aca73c2449297a0319(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[150], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d7be639a2f39c26098cbb9d5f2247fda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[86970], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d7be639a2f39c26098cbb9d5f2247fda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[86970], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b00cebb4afa1be06185c7c244065d210(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[242991], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b00cebb4afa1be06185c7c244065d210(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[242991], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_12658cf1c3e59813bcdfab500ea9e843(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[40], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_19da5067fb03899b27ffeaf39ee13c46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_86c8dc2666f0a25531ced25690d379d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[220968], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_86c8dc2666f0a25531ced25690d379d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[220968], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_aed3d7d935664c666fd48fb955c751c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[153450], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_aed3d7d935664c666fd48fb955c751c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[153450], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_665caa78c73eb7d1a1ffbcc77dd5153b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ca6daa6b6602ca1b52d093d1e0c07c7
        def get_inputs(self):
            return [
                paddle.uniform([3, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f5c25dcc0198f2ebb21d88f8991c9b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185691], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_2f5c25dcc0198f2ebb21d88f8991c9b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185691], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_25bd60a02307bb0e709db6d34b6b3d87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[113061], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_25bd60a02307bb0e709db6d34b6b3d87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[113061], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6da79125a5125270b49eecf31e3b3a0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6da79125a5125270b49eecf31e3b3a0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_275363c30eb80930f6ab19437e29c649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[205923], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_275363c30eb80930f6ab19437e29c649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[205923], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_725d0f28087b3ae525fac4fed3e39050(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[2204], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9fb25636c6b0d65806f236343802dc5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[123783], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9fb25636c6b0d65806f236343802dc5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[123783], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a353172979bf388fd2db7e98d749830d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[171888], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a353172979bf388fd2db7e98d749830d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[171888], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ea07b997cb13dc057aa05ee450ba2ec6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_425540fdeb361e1c9bc1a39e243ed555(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[551], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c37dbc2640398d098a134dbefaa024cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[217413], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c37dbc2640398d098a134dbefaa024cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[217413], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d95f601cc73b8a54e9c0c1b91bb66ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7840b8524711b7381dea3d5ddc25b8ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_bc3a79c58f61b3af180cec0fabc28819(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[8816], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_cfb995eadae953c6933cd760fe7f6bd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ca6daa6b6602ca1b52d093d1e0c07c7
        def get_inputs(self):
            return [
                paddle.uniform([6, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2733ff9809a087db290ce969f1cc9bc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ca6daa6b6602ca1b52d093d1e0c07c7
        def get_inputs(self):
            return [
                paddle.uniform([2, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d95f601cc73b8a54e9c0c1b91bb66ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_19da5067fb03899b27ffeaf39ee13c46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7840b8524711b7381dea3d5ddc25b8ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ea07b997cb13dc057aa05ee450ba2ec6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0488837e9206f26b5e76c56ecb5965cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185658], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0488837e9206f26b5e76c56ecb5965cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49e7416ebe6e41d68d8c1565980128d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185658], dtype='int32'), 'bool'),
            ]


    

if __name__ == '__main__':
    unittest.main()