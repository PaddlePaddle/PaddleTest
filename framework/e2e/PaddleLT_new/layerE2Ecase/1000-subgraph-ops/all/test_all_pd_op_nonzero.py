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
    class PrimitiveOp_46e9bb773ee13e640fff15e3cc565626(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_633f4494817a4e8ec605645f32ab484c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46e9bb773ee13e640fff15e3cc565626
        def get_inputs(self):
            return [
                paddle.uniform([4, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_954fcce5a82f4456adfe4668becf8009(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7d499e17ba3c60458ce3af1d99b98bfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_65caad2e75259a844a196a1d934414e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[150], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a30da74baae13462de6060927134e62d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[86970], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a30da74baae13462de6060927134e62d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[86970], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b8a0373b2158a9eecc5242659e93342c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[242991], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b8a0373b2158a9eecc5242659e93342c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[242991], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5a93776dc87db46d9c6b644d99228a4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[40], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7d499e17ba3c60458ce3af1d99b98bfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9228be00a5117ba88c813599031e9e22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[220968], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9228be00a5117ba88c813599031e9e22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[220968], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_31f2e1239604810e85eed5251aad93dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[153450], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_31f2e1239604810e85eed5251aad93dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[153450], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_3d761948886462f7e81f2b11e184f6a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46e9bb773ee13e640fff15e3cc565626
        def get_inputs(self):
            return [
                paddle.uniform([3, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22adbf09b9b8d8ee439b457d346de981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185691], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_22adbf09b9b8d8ee439b457d346de981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185691], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f1c7588653a3226a5beec0b64f87136d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[113061], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f1c7588653a3226a5beec0b64f87136d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[113061], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_576c58d4f5cdf55d0cf6b3f23202ae5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_576c58d4f5cdf55d0cf6b3f23202ae5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_964ccd94242ffe787be687c64e30a9c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[205923], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_964ccd94242ffe787be687c64e30a9c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[205923], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_90d343148bb76c648d0c8b0851f47765(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[2204], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_84d03d34f5d31d428280290a347cae63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[123783], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_84d03d34f5d31d428280290a347cae63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[123783], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1f373085ba802a00353287d9d387739f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[171888], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1f373085ba802a00353287d9d387739f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[171888], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_75e7c21cfbdc6a5a1d745df2196a2770(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_71c9cd902d59047f3c1e3a090ba21be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[551], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f9df362cda8ed5e5a2b029653102e2e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[217413], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f9df362cda8ed5e5a2b029653102e2e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[217413], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_416c9d1ef23dec56921f00fb48cb1e11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_dd7a8c72e41e5a575fdb1faf1dcbe8c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_56e0e0d45a360b051badb42072c5908f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[8816], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_42b312bccc31f32a2df1ef8e4fd8be68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46e9bb773ee13e640fff15e3cc565626
        def get_inputs(self):
            return [
                paddle.uniform([6, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d9db2b502feb3f2d080ae609a102231(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46e9bb773ee13e640fff15e3cc565626
        def get_inputs(self):
            return [
                paddle.uniform([2, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_416c9d1ef23dec56921f00fb48cb1e11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7d499e17ba3c60458ce3af1d99b98bfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_dd7a8c72e41e5a575fdb1faf1dcbe8c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_75e7c21cfbdc6a5a1d745df2196a2770(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5239da91a6eb431818bfb0209014668d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185658], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5239da91a6eb431818bfb0209014668d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185658], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_e2a58f4819373cfce08643ccec4c4632(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 80, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_29941132a1a602a18e48a2a95361f6ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2a58f4819373cfce08643ccec4c4632
        def get_inputs(self):
            return [
                paddle.uniform([4, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2b15379bc399a19dd156e467fc1de2d2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3800], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ba8bcd6dba9846c65b28c7a8e561331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b15379bc399a19dd156e467fc1de2d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_a1afb0697c65e03752681835c86bad43(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[150], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a3f52db52ded2da69644f06e9ee2a57f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a1afb0697c65e03752681835c86bad43
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[150], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_b42d07cfb769dde4e89b00fcff86c649(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86970], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d067a9744cbe788ae15bf854858b9425(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b42d07cfb769dde4e89b00fcff86c649
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[86970], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d067a9744cbe788ae15bf854858b9425(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b42d07cfb769dde4e89b00fcff86c649
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[86970], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_5f1e1e8a6a57fc2458fa6eb0878a971c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[242991], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a67e26c8553b34d04f45c4704d84d0c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f1e1e8a6a57fc2458fa6eb0878a971c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[242991], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a67e26c8553b34d04f45c4704d84d0c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f1e1e8a6a57fc2458fa6eb0878a971c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[242991], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_ff9c892632191d6d9692e92e489fecf2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_16efd3b865503b24970139250fc3e0bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff9c892632191d6d9692e92e489fecf2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[40], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6ba8bcd6dba9846c65b28c7a8e561331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b15379bc399a19dd156e467fc1de2d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_295de71910413d08b0d24d1a8abaa5d4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[220968], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4bbf00a61caa1f63f5424f3f2f3aa1ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_295de71910413d08b0d24d1a8abaa5d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[220968], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_4bbf00a61caa1f63f5424f3f2f3aa1ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_295de71910413d08b0d24d1a8abaa5d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[220968], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_ea6b7a2f15bbcd3dab2b8de09a3f8414(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[153450], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_970d907239798b0ae8a54e7638d79eab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea6b7a2f15bbcd3dab2b8de09a3f8414
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[153450], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_970d907239798b0ae8a54e7638d79eab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea6b7a2f15bbcd3dab2b8de09a3f8414
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[153450], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_6f783befd5839144e41f9cf327dfdf56(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3, 80, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ac517bb2896f590746d5e34368cc1589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f783befd5839144e41f9cf327dfdf56
        def get_inputs(self):
            return [
                paddle.uniform([3, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_972bc8be3d21fcd0ba27d18f3b2ce599(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[185691], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_156dd9190df273586102202a41a96d8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_972bc8be3d21fcd0ba27d18f3b2ce599
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185691], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_156dd9190df273586102202a41a96d8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_972bc8be3d21fcd0ba27d18f3b2ce599
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185691], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_97e81eebc0aa98317004c73454f8332a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[113061], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c172486616327c59ed72e5d8f4a9b787(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97e81eebc0aa98317004c73454f8332a
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[113061], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c172486616327c59ed72e5d8f4a9b787(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97e81eebc0aa98317004c73454f8332a
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[113061], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_8349c03b8ee4f501a410e70224036a60(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[15200], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_55c3b98c2bd544b4864c593bcd965d0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8349c03b8ee4f501a410e70224036a60
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_55c3b98c2bd544b4864c593bcd965d0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8349c03b8ee4f501a410e70224036a60
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_8ae824fb756f1cd0b7e1c805fd3174e0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[205923], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_43d324c61e24d34c474e39a68f0edc54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ae824fb756f1cd0b7e1c805fd3174e0
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[205923], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_43d324c61e24d34c474e39a68f0edc54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ae824fb756f1cd0b7e1c805fd3174e0
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[205923], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_a244d9ba24a896094f80dc6f540a7e6f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2204], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d44e73934307033cc94c65c7fe4e7f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a244d9ba24a896094f80dc6f540a7e6f
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[2204], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_694caa4535a9fdb15606c56666be1bec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[123783], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c21952bf529ee5eb9653615da7d9321(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_694caa4535a9fdb15606c56666be1bec
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[123783], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0c21952bf529ee5eb9653615da7d9321(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_694caa4535a9fdb15606c56666be1bec
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[123783], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_067e3ef38a9b6d35a2a8180f4b7eb48e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171888], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_213268312552e1670631e9cc9a080fd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_067e3ef38a9b6d35a2a8180f4b7eb48e
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[171888], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_213268312552e1670631e9cc9a080fd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_067e3ef38a9b6d35a2a8180f4b7eb48e
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[171888], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_4612126f04e11f4a72ec77008ca199fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[70], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_88cecf1b18a9acde5adea37dc3d02c1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4612126f04e11f4a72ec77008ca199fd
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_1a0f4b57d565170ca836277a2ff8e8c1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[551], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_890d7442d6a4a4c8393779ad1f338c30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a0f4b57d565170ca836277a2ff8e8c1
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[551], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_ae4a2a2cb057ffd8ef51843ed8977625(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[217413], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fbe9060be479666f1822127ce3f4d40c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae4a2a2cb057ffd8ef51843ed8977625
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[217413], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_fbe9060be479666f1822127ce3f4d40c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae4a2a2cb057ffd8ef51843ed8977625
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[217413], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_9019fec5b7dcbb6f5a2e3dbe7ee4f319(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[247], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c24b775dce64e428bdec36432e42eae0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9019fec5b7dcbb6f5a2e3dbe7ee4f319
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_446ed39270fdfe95fabed6c32a0c13fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[950], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9fb62472bf78ebfe126bb30e7df316e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_446ed39270fdfe95fabed6c32a0c13fc
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_ddd1dc2ee9e90ec21326f35b9e51256f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8816], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ff89cca9ac2cb4aa0b8d8617fe7580f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ddd1dc2ee9e90ec21326f35b9e51256f
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[8816], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_4056132b0a6aa89034084ecd78395170(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 80, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a5f93852f6f846a89925b2d00b4bc333(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4056132b0a6aa89034084ecd78395170
        def get_inputs(self):
            return [
                paddle.uniform([6, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_db16896245784ea323f80e8d5d0531e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 80, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_95278daba0f292b47137674fee97bf7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db16896245784ea323f80e8d5d0531e5
        def get_inputs(self):
            return [
                paddle.uniform([2, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c24b775dce64e428bdec36432e42eae0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9019fec5b7dcbb6f5a2e3dbe7ee4f319
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6ba8bcd6dba9846c65b28c7a8e561331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b15379bc399a19dd156e467fc1de2d2
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9fb62472bf78ebfe126bb30e7df316e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_446ed39270fdfe95fabed6c32a0c13fc
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_88cecf1b18a9acde5adea37dc3d02c1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4612126f04e11f4a72ec77008ca199fd
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_0df93a40cc313392fdb4e889e04cd36c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[185658], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8e0c5085a57f38e9468add1203f828d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0df93a40cc313392fdb4e889e04cd36c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185658], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b8e0c5085a57f38e9468add1203f828d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0df93a40cc313392fdb4e889e04cd36c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185658], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_209057e65fc85df7f98c0a9c3e27e5cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.nonzero(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9447a93a9d0a1b459ff87b72b5b5dc75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_209057e65fc85df7f98c0a9c3e27e5cc
        def get_inputs(self):
            return [
                paddle.uniform([4, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d499e17ba3c60458ce3af1d99b98bfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_65caad2e75259a844a196a1d934414e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[150], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a30da74baae13462de6060927134e62d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[86970], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a30da74baae13462de6060927134e62d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[86970], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b8a0373b2158a9eecc5242659e93342c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[242991], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b8a0373b2158a9eecc5242659e93342c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[242991], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5a93776dc87db46d9c6b644d99228a4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[40], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7d499e17ba3c60458ce3af1d99b98bfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9228be00a5117ba88c813599031e9e22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[220968], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9228be00a5117ba88c813599031e9e22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[220968], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_31f2e1239604810e85eed5251aad93dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[153450], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_31f2e1239604810e85eed5251aad93dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[153450], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0bef2f70e4ae01a48ef1ebf1262324d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_209057e65fc85df7f98c0a9c3e27e5cc
        def get_inputs(self):
            return [
                paddle.uniform([3, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22adbf09b9b8d8ee439b457d346de981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185691], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_22adbf09b9b8d8ee439b457d346de981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185691], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f1c7588653a3226a5beec0b64f87136d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[113061], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f1c7588653a3226a5beec0b64f87136d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[113061], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_576c58d4f5cdf55d0cf6b3f23202ae5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_576c58d4f5cdf55d0cf6b3f23202ae5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_964ccd94242ffe787be687c64e30a9c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[205923], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_964ccd94242ffe787be687c64e30a9c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[205923], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_90d343148bb76c648d0c8b0851f47765(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[2204], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_84d03d34f5d31d428280290a347cae63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[123783], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_84d03d34f5d31d428280290a347cae63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[123783], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1f373085ba802a00353287d9d387739f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[171888], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1f373085ba802a00353287d9d387739f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[171888], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_75e7c21cfbdc6a5a1d745df2196a2770(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_71c9cd902d59047f3c1e3a090ba21be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[551], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f9df362cda8ed5e5a2b029653102e2e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[217413], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f9df362cda8ed5e5a2b029653102e2e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[217413], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_416c9d1ef23dec56921f00fb48cb1e11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_dd7a8c72e41e5a575fdb1faf1dcbe8c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_56e0e0d45a360b051badb42072c5908f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[8816], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f21014c0eaf97284f5d3c494a6de79de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_209057e65fc85df7f98c0a9c3e27e5cc
        def get_inputs(self):
            return [
                paddle.uniform([6, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_731de5be30d6fcfbfbdde3eff5fef539(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_209057e65fc85df7f98c0a9c3e27e5cc
        def get_inputs(self):
            return [
                paddle.uniform([2, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_416c9d1ef23dec56921f00fb48cb1e11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7d499e17ba3c60458ce3af1d99b98bfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_dd7a8c72e41e5a575fdb1faf1dcbe8c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_75e7c21cfbdc6a5a1d745df2196a2770(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5239da91a6eb431818bfb0209014668d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185658], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5239da91a6eb431818bfb0209014668d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_954fcce5a82f4456adfe4668becf8009
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[185658], dtype='int32'), 'bool'),
            ]


    

if __name__ == '__main__':
    unittest.main()