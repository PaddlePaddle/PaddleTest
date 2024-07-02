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
    class PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3af3f1ba388e8934254c85a1edb5490b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_408e0e6f9c13790c572948a7e9c427f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[150], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_1a8f13016361794a596aeb4d78448be1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ca61d754d6a18aff1cce17ba29ecb6df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_fc39d7f4ac33f80664c8d2513dd4d622(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_add53d9d44122c1969f20ea77196a9ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3af3f1ba388e8934254c85a1edb5490b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6d6717b4b07587f9e36c1050e6e093ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c13b93d0028ef423d8d08fc13c11d6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6eac218e7d3954f748ccae29dd2c1f77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1043921e32a2f83d42529c50caf1788f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_972fabb404dbd82af4ea99e7b7ffa181(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e8c164b38d7a7c3339c14046955369c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f6d75f6844cd4115f3c6774c3fbf1996(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cbdf3e97cc7d493bbb702a19a8bac791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_cbdf3e97cc7d493bbb702a19a8bac791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_1043921e32a2f83d42529c50caf1788f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_70850e149721ef0d221b8a0386e0e750(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_70850e149721ef0d221b8a0386e0e750(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0b930d8b486cd50945d47278c2814dda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a7cca6fdefaea6c0cb9068220ff438b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2413cd0d0302676f07731c30ccaac868(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2204], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_2867d1bd31f67115b4da224005203ea5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ca3d774e6d9f63d008fa77d9ff52062a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f83760911f49cf8bf877f3d580262a5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b236d6932cabec59a887f0f2d58302f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[551], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ccc99662beeebde73dd60c0d558a466f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_697079c78de8b436a1518284ca1eaeed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_29b125820557c9eb96f3cdc162a3cbcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_526c36dd9764af5c74cc5306e4a20987(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[8816], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_e8c164b38d7a7c3339c14046955369c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e8c164b38d7a7c3339c14046955369c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e8c164b38d7a7c3339c14046955369c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1bfb2630a57ab63a63fdd3b901112fdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e8c164b38d7a7c3339c14046955369c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e8c164b38d7a7c3339c14046955369c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e8c164b38d7a7c3339c14046955369c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1bfb2630a57ab63a63fdd3b901112fdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0b930d8b486cd50945d47278c2814dda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_70850e149721ef0d221b8a0386e0e750(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_697079c78de8b436a1518284ca1eaeed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3af3f1ba388e8934254c85a1edb5490b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_29b125820557c9eb96f3cdc162a3cbcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_e8c164b38d7a7c3339c14046955369c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f83760911f49cf8bf877f3d580262a5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7ee5e6c26a53c9705c8aefc9c00df2e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e8c164b38d7a7c3339c14046955369c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0b930d8b486cd50945d47278c2814dda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcb89634b698a3ef2ac20e72e6f6b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3af3f1ba388e8934254c85a1edb5490b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_408e0e6f9c13790c572948a7e9c427f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[150], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ca61d754d6a18aff1cce17ba29ecb6df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_fc39d7f4ac33f80664c8d2513dd4d622(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_add53d9d44122c1969f20ea77196a9ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3af3f1ba388e8934254c85a1edb5490b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6d6717b4b07587f9e36c1050e6e093ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c13b93d0028ef423d8d08fc13c11d6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6eac218e7d3954f748ccae29dd2c1f77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_17c2b72d667da5a48b1e41f8781c2bbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_972fabb404dbd82af4ea99e7b7ffa181(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_98e3428d485a3c45bfb4d3d58ad56707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a1d5c10d2ca3f232301d6c26cdeaace2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cbdf3e97cc7d493bbb702a19a8bac791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_cbdf3e97cc7d493bbb702a19a8bac791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_17c2b72d667da5a48b1e41f8781c2bbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_27c8e54cea5038665aa5627f654bf685(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_27c8e54cea5038665aa5627f654bf685(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_dddad5686356fa55a739014c9218ee61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a7cca6fdefaea6c0cb9068220ff438b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2413cd0d0302676f07731c30ccaac868(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2204], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_2867d1bd31f67115b4da224005203ea5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ca3d774e6d9f63d008fa77d9ff52062a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f83760911f49cf8bf877f3d580262a5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b236d6932cabec59a887f0f2d58302f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[551], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ccc99662beeebde73dd60c0d558a466f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_697079c78de8b436a1518284ca1eaeed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_29b125820557c9eb96f3cdc162a3cbcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_526c36dd9764af5c74cc5306e4a20987(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[8816], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_98e3428d485a3c45bfb4d3d58ad56707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_98e3428d485a3c45bfb4d3d58ad56707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_98e3428d485a3c45bfb4d3d58ad56707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_af6936807b114c5ab242cb23daaa0cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_98e3428d485a3c45bfb4d3d58ad56707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_98e3428d485a3c45bfb4d3d58ad56707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_98e3428d485a3c45bfb4d3d58ad56707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_af6936807b114c5ab242cb23daaa0cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_dddad5686356fa55a739014c9218ee61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_27c8e54cea5038665aa5627f654bf685(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_697079c78de8b436a1518284ca1eaeed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3af3f1ba388e8934254c85a1edb5490b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_29b125820557c9eb96f3cdc162a3cbcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_98e3428d485a3c45bfb4d3d58ad56707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f83760911f49cf8bf877f3d580262a5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad80a8a49caefeed292cda7e338bdac
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7ee5e6c26a53c9705c8aefc9c00df2e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8f13016361794a596aeb4d78448be1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_98e3428d485a3c45bfb4d3d58ad56707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_dddad5686356fa55a739014c9218ee61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e214cbad29e15e06eef0f9ad49f72913
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()