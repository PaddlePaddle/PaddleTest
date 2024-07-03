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
            PADDLE_DEBUG_ENABLE_CINN=False,
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
    PADDLE_DEBUG_CINN_STAGE_NAME="backend",
    PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF=False,
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





last_stage_failed = (IsCinnStageEnableDiff() and LastCINNStageFailed())
class PrimitiveOp_af8a58da019b9911d1523aab76aa63cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1a477b5f81c5624542bea00bf1e5f09a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_23a8cccffbfadc370a9e40378225cc65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc52cae1dcd6cc3b092b9a6aa63253f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc52cae1dcd6cc3b092b9a6aa63253f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_c532038966272a76e2f0a3f650179278(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9c3f3f9b6cf285d240eaf1fe43ce69d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c532038966272a76e2f0a3f650179278
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_eead0fa056a7724d8d908e638ee1c41d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_904cf5a3eedf70d8215b34b234ddf439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eead0fa056a7724d8d908e638ee1c41d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_b9c929d2f0698f14085485ef8987b832(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5974c8ca9f9ed2c55e4d2684f9b5625a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9c929d2f0698f14085485ef8987b832
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_89f85757158a2321ad582f23efd29c83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf506d7e017960aa8d74e16bdde5beef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89f85757158a2321ad582f23efd29c83
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 152, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_a70976acb84d41295856a0b44477c400(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4115fdf124f8a56d40b966f2069e3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a70976acb84d41295856a0b44477c400
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_9e7615c69dd71d274dc223ad336df9c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f7525708d844233b7910154546ac134f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e7615c69dd71d274dc223ad336df9c9
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[0.0]]], [[[0.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]



class PrimitiveOp_13c3663f29586eae9593bac13364d010(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_279f9fb1c8381f0d18e2f57b9f675217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13c3663f29586eae9593bac13364d010
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_9adcd3bf70b160b0687577265b910a4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_722cd5f9f86454bf5918d3900c66caea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9adcd3bf70b160b0687577265b910a4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_f782d7dbea763a914af953ed3052afdd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efa4cade74a59784ea46c53b12adf8ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_d76cab07479f76fa0c11ecb458cdbfe8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_366b90c26c22c2b01e7ff9eda7bb5bd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76cab07479f76fa0c11ecb458cdbfe8
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0087d5b9cc883c3df59d7dbe039f0bc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e7615c69dd71d274dc223ad336df9c9
    def get_inputs(self):
        return [
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_aae5c56eace724340840f8107f1c620d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6ac30c019f10474607494e73181de661(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18265779316425323], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_654dd01b30b395fd9e82ea165c68c50c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_6bbfcb13635d0ee466e6ef38a6192283(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1ea2bbe041faa89175f4de5a200c8edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bbfcb13635d0ee466e6ef38a6192283
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_20e7dc14525835bfca2e3af432167a2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cff5ed252f0a59b0c497fed595c414b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cff5ed252f0a59b0c497fed595c414b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cff5ed252f0a59b0c497fed595c414b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_cdd667a1c8b27e86a760d9efef6c29bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7542cd2f425ded9ca02642c3420986af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd667a1c8b27e86a760d9efef6c29bc
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9687598f80eb1b3665c885e1eb1ff5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d61564a854d48b3c32d0bba1c1804d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4c8b19199ddf4f084ed43cb3e70f61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]



class PrimitiveOp_c3da4619f9159da19973072cca96bf1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b17cacbbe376fdbb6aac0db868a8ca33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3da4619f9159da19973072cca96bf1f
    def get_inputs(self):
        return [
            paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.33804893493652344]], [[0.39942264556884766]], [[0.09662508219480515]], [[0.1508401781320572]]], dtype='float32').reshape([4, 1, 1]),
        ]



class PrimitiveOp_099ad7c631c3a89cfec948b4a0ec88f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_564cb00758dd0dfcda9a9a4fe3c6b9b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_099ad7c631c3a89cfec948b4a0ec88f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ac0ae25d3f54746d7b9bb8b633be03e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_722b282bf145a43366827a8915b920e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac0ae25d3f54746d7b9bb8b633be03e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4b0569bac51415dd1bf6a7eac557d6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_921c017be1674eff158170d0ab1bd3dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 1, 9, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 2, 16, 9, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8902975d9ae036c8ac5b73b02eea4601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_921c017be1674eff158170d0ab1bd3dd
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 1, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8fd3d9e38493f516d3f8dd997e64ad9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bbfcb13635d0ee466e6ef38a6192283
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7a246b9a0b3b194af729513f492a7089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.40364521741867065], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe916c13b8d42e302b3d080a2cab9f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe916c13b8d42e302b3d080a2cab9f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe916c13b8d42e302b3d080a2cab9f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2bc2118009851a836ce0d57314b82efb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b03b035b4bf4982d10aa68418a87052b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bc2118009851a836ce0d57314b82efb
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.9297406077384949]], [[0.8804839849472046]], [[0.9003913998603821]], [[0.9182154536247253]], [[0.9530584812164307]], [[0.9100152850151062]], [[0.9088941812515259]], [[0.9383436441421509]], [[0.9679425358772278]], [[0.9257610440254211]], [[0.9264084100723267]], [[0.92100989818573]], [[0.9689187407493591]], [[0.8751189708709717]], [[0.9151067733764648]], [[0.8847223520278931]], [[0.9026293158531189]], [[0.9414359927177429]], [[0.8382887244224548]], [[0.9695025086402893]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]



class PrimitiveOp_c13876966c9e62bb2b8d3893967019ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0519321d62aa40fa2cd6b0540ea465bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c13876966c9e62bb2b8d3893967019ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_843c795c904b4dd89d0f9575184e65d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_85aad4baa86317c13730ec2746239704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_843c795c904b4dd89d0f9575184e65d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_934e944ba9c81a577462dce5e614d0d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b687d2d7e85fb1eb29bef5016c4baaa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_934e944ba9c81a577462dce5e614d0d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_685215e39df696ab66134852db25f88e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_078bc0e96d6acde55acc91ca459d8dc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc1117342399bb3c7b22e585664f0caa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc1117342399bb3c7b22e585664f0caa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_cd9cbd8acbc7f845c7de45667aa7f49e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_896825b4b1c9a05cf54abd1af06d8539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd9cbd8acbc7f845c7de45667aa7f49e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24692094326019287]]], dtype='float32').reshape([1, 1, 1]),
        ]



class PrimitiveOp_6dca5adf2513d57655fa0032db9d3138(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2100, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_818e3b8b3033c95930e9a783315b22c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dca5adf2513d57655fa0032db9d3138
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_85e179cd87509481639fbf2c2186f445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18271088600158691], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3f8ff48b7afabae7e9478bcac8ec66e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_07bf753e32b01b76520d7697121f07ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4541bd25202b745afa301d2ba210f189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_00f23910479f0f2ee0a1f0a25caaf8bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d06bcb6e2ab7934d11a685afdefd1401(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00f23910479f0f2ee0a1f0a25caaf8bf
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_25580fa142b192c65f743c2d3584165c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8d3fc465ee6b8e783fa4da8b322cdb94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25580fa142b192c65f743c2d3584165c
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f60c667b0241e05e213057e568f47a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f60c667b0241e05e213057e568f47a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f60c667b0241e05e213057e568f47a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f60c667b0241e05e213057e568f47a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_92ef1838f01dabe47d9d4d923bd2f52d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_674cf7ec3cc9630b256f7a9c2538487b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_674cf7ec3cc9630b256f7a9c2538487b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_674cf7ec3cc9630b256f7a9c2538487b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1785e361166271bb0dfade7f72a30966(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac0ae25d3f54746d7b9bb8b633be03e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cf09903dd7da747e167d851a4c22ed00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_934e944ba9c81a577462dce5e614d0d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_af833cfa6e363039d7484e2348ab7942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_76fec3bf88c0e7b9b52efe4c6677bfbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1152, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1152, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b936accbab46a866d76fd1d58ffcfbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76fec3bf88c0e7b9b52efe4c6677bfbc
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_3d5112d9c946203be8f957691d36cce8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62bda368160760e7237e5b9dd732db69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55af99991338ff59c8fecabc472ae70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_843c795c904b4dd89d0f9575184e65d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8a7e4f6d3368128977dfc09b4de0559b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25580fa142b192c65f743c2d3584165c
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e6a656541f0f435fab771c6ed7e41840(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 68, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cd5bb4829b08b061c07ccca190570738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6a656541f0f435fab771c6ed7e41840
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.11957564949989319], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_faa2fda39bec88497a3e34a8cf9419c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_744a74dc103d74f4c2cb98f743ce91dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c5735b185675bfb753fbaa5c36280af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd667a1c8b27e86a760d9efef6c29bc
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1fbda4d43d58029cd139c27e0e7f82a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e63e2b01e71e8b74fe95505746636016(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.284286230802536], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a3d2686466336934f3b891cc892e4d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_7bf5744d1617a346b22175977c14d76a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7572de5a2bc518c42c19bfbade18022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bf5744d1617a346b22175977c14d76a
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.761617124080658]], [[0.7681009769439697]], [[0.7132568359375]], [[0.7124687433242798]], [[0.6587449312210083]], [[0.7245516777038574]], [[0.8148982524871826]], [[0.6684139966964722]], [[0.6617448925971985]], [[0.5804407596588135]], [[0.7869804501533508]], [[0.7067265510559082]], [[0.6869311928749084]], [[0.7316954731941223]], [[0.780451774597168]], [[0.8454877138137817]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_83731a5aa1162126a53e5236f31f9d44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a521411af465f3006400f4d8526f366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd667a1c8b27e86a760d9efef6c29bc
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_254f22656f03131bb87e84457a937586(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 44, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 44, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1f4fd6f313c3a577b9b72fb2d4e581f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_254f22656f03131bb87e84457a937586
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_21bca7d2ea0792c723636d9d50f33db8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a70976acb84d41295856a0b44477c400
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_02b9d20c36611ef2fad29beb0243d068(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0816437982f8bc924e2e41609e48669e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[-0.34831881523132324, -0.3267747759819031]], [[-0.4283560514450073, 0.12656813859939575]], [[0.40908509492874146, -0.2860028147697449]], [[0.0807693600654602, -0.30509230494499207]], [[-0.2860870063304901, 0.05865928530693054]], [[0.35354083776474, -0.2900979816913605]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_419345efde6fd279a358e98f49e39cab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[-0.034599900245666504, -0.1461290866136551]], [[-0.1180865466594696, 0.09195978194475174]], [[-0.041482940316200256, 0.022733867168426514]], [[0.07443660497665405, -0.2604130208492279]], [[-0.28827643394470215, -0.28109776973724365]], [[-0.09692291915416718, -0.1548452377319336]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ef02a1a97925635403fba8a445ebf9cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.34831881523132324, -0.3267747759819031]], [[-0.4283560514450073, 0.12656813859939575]], [[0.40908509492874146, -0.2860028147697449]], [[0.0807693600654602, -0.30509230494499207]], [[-0.2860870063304901, 0.05865928530693054]], [[0.35354083776474, -0.2900979816913605]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[-0.34831881523132324, -0.3267747759819031]], [[-0.4283560514450073, 0.12656813859939575]], [[0.40908509492874146, -0.2860028147697449]], [[0.0807693600654602, -0.30509230494499207]], [[-0.2860870063304901, 0.05865928530693054]], [[0.35354083776474, -0.2900979816913605]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c14de18fd199000d25ebc482fc962205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.034599900245666504, -0.1461290866136551]], [[-0.1180865466594696, 0.09195978194475174]], [[-0.041482940316200256, 0.022733867168426514]], [[0.07443660497665405, -0.2604130208492279]], [[-0.28827643394470215, -0.28109776973724365]], [[-0.09692291915416718, -0.1548452377319336]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[-0.034599900245666504, -0.1461290866136551]], [[-0.1180865466594696, 0.09195978194475174]], [[-0.041482940316200256, 0.022733867168426514]], [[0.07443660497665405, -0.2604130208492279]], [[-0.28827643394470215, -0.28109776973724365]], [[-0.09692291915416718, -0.1548452377319336]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a069223f9e76b2acc622fd31ab8243da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd9cbd8acbc7f845c7de45667aa7f49e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.10894569754600525], [0.08911313861608505], [0.12436171621084213], [0.03143560141324997], [0.02490701898932457], [0.09564900398254395]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[0.4026823937892914], [0.07231664657592773], [0.011129551567137241], [0.07582680135965347], [0.40325236320495605], [0.13427451252937317]]], dtype='float32').reshape([1, 6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_31746eb951ecb795b49e2414d0c69421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd9cbd8acbc7f845c7de45667aa7f49e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0033864504657685757], [0.003352757077664137], [0.00010585029667709023], [0.019867869094014168], [0.06527575105428696], [0.006096151191741228]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[0.4026823937892914], [0.07231664657592773], [0.011129551567137241], [0.07582680135965347], [0.40325236320495605], [0.13427451252937317]]], dtype='float32').reshape([1, 6, 1]),
        ]



class PrimitiveOp_14d5f61246d6ea1a7743b7e03031aed2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f86924b63a01345b69f1bc86b00cb9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14d5f61246d6ea1a7743b7e03031aed2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_7035d2a1d23c8fb26271d4afbe956f92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 1, 49, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4, 16, 49, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_10497d1e7c099898200c21894602bcf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7035d2a1d23c8fb26271d4afbe956f92
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 1, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_906d96c087b742fe288b19f09f532ecc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_934e944ba9c81a577462dce5e614d0d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0cc897a4707e64d582639515069ee512(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25580fa142b192c65f743c2d3584165c
    def get_inputs(self):
        return [
            paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bb952a2323fb83d686cfbf3dfd528ada(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_053d5a835a5aaf0a0b4c69f5027f45a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bbfcb13635d0ee466e6ef38a6192283
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2e63809b6d25153847609929fc5061a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29e04ac46758bf3ab7f17e1dbe67580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29e04ac46758bf3ab7f17e1dbe67580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29e04ac46758bf3ab7f17e1dbe67580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29e04ac46758bf3ab7f17e1dbe67580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ff6eacf1697f2759b52440356cb62dc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e7615c69dd71d274dc223ad336df9c9
    def get_inputs(self):
        return [
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_94f04d32a17310775c93c2bc96774a21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_01807582e82a245cb29bb320e27da5c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94f04d32a17310775c93c2bc96774a21
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_76127761220767ab72012abe4e5540b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1523a8a727bf3d522a0f9da6844e3700(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.34787654876709, 2.159102201461792, 1.9213602542877197, 2.2024269104003906, 2.1114273071289062, 2.1365914344787598, 1.959598422050476, 1.8337061405181885, 1.8550523519515991, 1.9595694541931152, 2.091510772705078, 2.2969653606414795, 1.895961880683899, 1.937199592590332, 2.1223039627075195, 2.2149817943573], dtype='float32').reshape([16]),
            paddle.to_tensor([0.6219276189804077, 0.8931844830513, 0.7872183322906494, 0.7686119079589844, 0.6709363460540771, 0.7716026306152344, 0.866905689239502, 0.9793623685836792, 0.9484323859214783, 0.5349599123001099, 0.8685113191604614, 0.550764799118042, 0.9109439849853516, 0.9845443964004517, 0.6136507987976074, 0.5252518057823181], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c67c60b88a4a77c53ede801e4df3d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.8714301586151123, 2.0343306064605713, 2.213071584701538, 2.2041175365448, 2.139944076538086, 2.1380159854888916, 1.9252699613571167, 2.2255287170410156, 1.8711376190185547, 2.096949577331543, 2.2395358085632324, 2.1725518703460693, 2.0819284915924072, 2.409768581390381, 2.231198310852051, 2.095301628112793], dtype='float32').reshape([16]),
            paddle.to_tensor([0.3780723512172699, 0.10681550204753876, 0.21278166770935059, 0.23138809204101562, 0.32906362414360046, 0.22839735448360443, 0.13309429585933685, 0.0206376351416111, 0.05156761035323143, 0.46504005789756775, 0.13148869574069977, 0.449235200881958, 0.08905600011348724, 0.015455592423677444, 0.3863491714000702, 0.4747481942176819], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68d9f7dfb97f6e7504f055d82d860fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5419362783432007, 0.5364436507225037, 0.49585777521133423, 0.5507045388221741, 0.5302027463912964, 0.5342292189598083, 0.4887573719024658, 0.46044811606407166, 0.46397048234939575, 0.505864143371582, 0.5277435779571533, 0.5602686405181885, 0.4781308174133301, 0.4861258566379547, 0.5410937666893005, 0.5395409464836121], dtype='float32').reshape([16]),
            paddle.to_tensor([0.368032842874527, 0.10307395458221436, 0.32535305619239807, 0.47889789938926697, 0.4783645570278168, 0.18116134405136108, 0.2584570348262787, 0.43919625878334045, 0.22095052897930145, 0.2511819303035736, 0.045693691819906235, 0.06585970520973206, 0.28214719891548157, 0.18147172033786774, 0.047013502568006516, 0.05115712434053421], dtype='float32').reshape([16]),
        ]



class PrimitiveOp_2ad48c54adf7e753113a4039ad9c4b79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93ab3f4935dd16f2cf6396bce3eff295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ad48c54adf7e753113a4039ad9c4b79
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eccc1737bd9930ba437237b60cc05ebb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.32124772667884827], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_f47f8cde93bde2463ad8c9966f0858bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0dc0def441400c82186027b3788fdb4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f47f8cde93bde2463ad8c9966f0858bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_982337694735f10f0b236a2b970ec976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_982337694735f10f0b236a2b970ec976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_982337694735f10f0b236a2b970ec976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62985af25a34b3f46dc96add26a4573b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2072802186012268], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_3dae0ef18cee31034ea41daebea35b37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 300, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5dccc4739ff4873046f74c8ef2189d59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3dae0ef18cee31034ea41daebea35b37
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.18305830657482147, 0.25759702920913696, 0.46923521161079407, 0.37422502040863037]]], dtype='float32').reshape([1, 1, 4]),
        ]



class PrimitiveOp_b721c68176da05878f6b79cb07de790f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7fb4080246c2f3047a7335235100319(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b721c68176da05878f6b79cb07de790f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c4d5816bc864c9f29facbeed82e4f659(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.47799554467201233], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9f599ba981938f2941a71c8da1f6d634(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.05867317318916321], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ef2b2eaacc74ac89912817709f7e01a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.029930606484413147], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cff5ed252f0a59b0c497fed595c414b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cff5ed252f0a59b0c497fed595c414b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cff5ed252f0a59b0c497fed595c414b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cff5ed252f0a59b0c497fed595c414b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31b48274dcd5e8f61211460521e2419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31b48274dcd5e8f61211460521e2419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31b48274dcd5e8f61211460521e2419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e0b1510d9db691deddad53cd48100fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_25f2ea09b0ea0f62b92732a237cbc195(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5f183204f850c4eb881cef35813b816b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25f2ea09b0ea0f62b92732a237cbc195
    def get_inputs(self):
        return [
            paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.2647590637207031]], [[0.05532848462462425]], [[0.46784508228302]]], dtype='float32').reshape([3, 1, 1]),
        ]



class PrimitiveOp_646be0e6d30aec875c17cadf40260b2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_94d2f5881c39b96ebb8a67f7169ba79b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_646be0e6d30aec875c17cadf40260b2a
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b2d3f5d0aac0982fb9a46ff33892f1a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4a96f0e8106719f0e89145e90636dae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00f23910479f0f2ee0a1f0a25caaf8bf
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efa4cade74a59784ea46c53b12adf8ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2fe744fcc00520d22cecdcf02dd8280a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6a656541f0f435fab771c6ed7e41840
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3176971971988678], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_15adfb1bc330804d561de35ffad6122d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c88bcc9fd6c99fcc41cbd6d9cfd6b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15adfb1bc330804d561de35ffad6122d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c942c2a837beb7f18cb07a5768554c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58f88baaa32f7560d5a577fc2aa27bc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bbfcb13635d0ee466e6ef38a6192283
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2ded4eb5408abf9cc641bb9821611450(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 320, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 320, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_01c7e4042232f042f6237f16e2661487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ded4eb5408abf9cc641bb9821611450
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2829c59a382391a54571ed732ee6e05a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_b885a9beea045cc51cf748579924e835(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 872, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb0a7c0397629d18b1efb8c62ad7f888(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b885a9beea045cc51cf748579924e835
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e39a887b78cb2460a1376fc32b9060cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e39a887b78cb2460a1376fc32b9060cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e39a887b78cb2460a1376fc32b9060cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e39a887b78cb2460a1376fc32b9060cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_7b21ff1ab3efead121219b2820c5d9d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bbefbf139817c59f11332e32cba2fb8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b21ff1ab3efead121219b2820c5d9d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82d05589d9ea0eb33bcfd1953178ecbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_934e944ba9c81a577462dce5e614d0d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6196dcdf0ca22219292cf0f99748f1a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6196dcdf0ca22219292cf0f99748f1a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6196dcdf0ca22219292cf0f99748f1a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6196dcdf0ca22219292cf0f99748f1a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6196dcdf0ca22219292cf0f99748f1a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_714d56a3218460f5481fdb9da264cf9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c5d5ca9955550f2d9fec6eb1883cf9f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c5d5ca9955550f2d9fec6eb1883cf9f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6196dcdf0ca22219292cf0f99748f1a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7f1a1ad7772d4e7eadfe08b75f184117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_951f02b2430f69986049c2f19ea36575(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e7615c69dd71d274dc223ad336df9c9
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_50334d8ed6da651e696a57e64f2b6393(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24196235835552216], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9beb45f30d453aeb33a3d05001aeaea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ce23797ea88c99211a39d63f06e5e000(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e7615c69dd71d274dc223ad336df9c9
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]



class PrimitiveOp_25a87951109d1a21d9596a75a3e62cba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6001132088ecf3084e1e08895742ea58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25a87951109d1a21d9596a75a3e62cba
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cdf61c2608ca99ea725afd0b8be12385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6a656541f0f435fab771c6ed7e41840
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1614248901605606], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_0fcf14b7481ea99595319c1717d48479(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 4, 1, 49, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 4, 16, 49, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b2e431f6fdacaae2dc75fb5b4040e0d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fcf14b7481ea99595319c1717d48479
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 1, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13b1588f6e9741223c08664a57d8df13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13b1588f6e9741223c08664a57d8df13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13b1588f6e9741223c08664a57d8df13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7af2d4174f5334475826f5d31b62840d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7af2d4174f5334475826f5d31b62840d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7af2d4174f5334475826f5d31b62840d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7af2d4174f5334475826f5d31b62840d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9cc14f96ffa75489a6d32616323091f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_843c795c904b4dd89d0f9575184e65d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_98800cffd56a5bc63e8434c6331a920f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd667a1c8b27e86a760d9efef6c29bc
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_802f57c2446a8ffcddb11015e0a2e02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aaa2cba263a3d3350a55cef1034139fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e7615c69dd71d274dc223ad336df9c9
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_da89849831834739a5f4bc6bcd0ec3b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9c9ed155a63eb9e327a09fa6948881b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da89849831834739a5f4bc6bcd0ec3b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c074625128d3b964108faa3bbf72d1be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b810afd60e0d6704e48d33abb5630f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ded4eb5408abf9cc641bb9821611450
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8eab8735b4cdf331ba7f8761e9afd2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8eab8735b4cdf331ba7f8761e9afd2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8eab8735b4cdf331ba7f8761e9afd2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b135ce9a450ba0418060cb7ba706b9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e61bab7fdd2055a521d09693a09a81ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c563f7109799979c252d5261ca67b188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ede582b6acf55247c0f8cdc42da718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ede582b6acf55247c0f8cdc42da718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ede582b6acf55247c0f8cdc42da718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ede582b6acf55247c0f8cdc42da718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_91da848df7d9cb3e62ee12ef1594efa6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_523718ed6b8a648363b64ac42c3590ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91da848df7d9cb3e62ee12ef1594efa6
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7a91746fcd10805819c2f5a43435c1c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0721592903137207], [0.07056476175785065], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f1fc2f0ecaeefef8ce76955680f1d34d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.023881256580352783], [-0.17093104124069214], [-0.040314070880413055], [-0.23770397901535034], [0.19680941104888916], [-0.1749243587255478], [0.09875815361738205], [-0.23626181483268738], [-0.24265314638614655]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.10350233316421509], [-0.380374550819397], [-0.2866537570953369], [0.0721592903137207], [0.15432409942150116], [0.2023364007472992], [-0.22068245708942413], [-0.19273780286312103], [-0.3567226529121399]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f69f1f189af735ddb92a4d261136e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.06902246177196503], [-0.22078408300876617], [0.07766935229301453], [0.012395694851875305], [-0.15766564011573792], [-0.1496490091085434], [-0.18517188727855682], [-0.3871796131134033], [-0.1858700066804886]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.3100380301475525], [0.2617378234863281], [0.2190100997686386], [0.3838931620121002], [0.2870804965496063], [-0.20615726709365845], [0.08606493473052979], [-0.0019951313734054565], [-0.11980006098747253]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b27cb8116776b259e84a410d46512fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.21349862217903137], [-0.0710458755493164], [0.10224482417106628], [0.018052708357572556], [0.19680941104888916], [-0.07336302101612091], [0.09875815361738205], [-0.22516174614429474], [-0.1858700066804886]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.0028449296951293945], [0.2617378234863281], [0.2190100997686386], [0.3838931620121002], [0.37083983421325684], [0.2023364007472992], [0.14159324765205383], [-0.0019951313734054565], [-0.11980006098747253]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1af23f47ff4a551e2cd3452c60285c76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac0ae25d3f54746d7b9bb8b633be03e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4683afac25a58ffc882d2d044f6965da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ad48c54adf7e753113a4039ad9c4b79
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d7eb67e4bbcc0792c2cd5755376dae4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f77deb4272df7b4fedf149f58b4b8af6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa10ef06d618964e716171173bb90296(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c532038966272a76e2f0a3f650179278
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2ed22ccee65afb4350547a5771b0998e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_623234c8b17eaf56626d804ba94bba60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed22ccee65afb4350547a5771b0998e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31835365295410156], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_623234c8b17eaf56626d804ba94bba60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed22ccee65afb4350547a5771b0998e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31835365295410156], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_07e1134c7135d09d326f33d240f511c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_30c1b0cb1d956d1d24f5e246de248449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07e1134c7135d09d326f33d240f511c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4ae5bb312abb4b2365e3e5e783f4a174(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2b7513130785f0a14a6f6a0a8a01b193(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ae5bb312abb4b2365e3e5e783f4a174
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09322686493396759], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_094ca6d6965676138e63dc62afb1fadf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13c3663f29586eae9593bac13364d010
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df465bbce7d02829ee5e9476b10971b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c532038966272a76e2f0a3f650179278
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cd65f15b9b3693502cb70f9ca87815d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a0ae9145ba7ee5195fb6f7dbe4491de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a0ae9145ba7ee5195fb6f7dbe4491de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a0ae9145ba7ee5195fb6f7dbe4491de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a0ae9145ba7ee5195fb6f7dbe4491de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a0ae9145ba7ee5195fb6f7dbe4491de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6818cd45849359dbd22f5e06e8fd79b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6818cd45849359dbd22f5e06e8fd79b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a0ae9145ba7ee5195fb6f7dbe4491de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_a10b9c0cb44b68461521ecd20c14a9f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3af9b98add13c4007e40a4e087812c66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a10b9c0cb44b68461521ecd20c14a9f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_154b3ce918a6a42469cb484e432c9e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c532038966272a76e2f0a3f650179278
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_db5613fc8c863d28c4c54ff6a1b6eea4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e7615c69dd71d274dc223ad336df9c9
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_702f930499f9d8b02b50b1db4c52b248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_702f930499f9d8b02b50b1db4c52b248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_702f930499f9d8b02b50b1db4c52b248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e891dfc97bb0762adf9bdedb1fd790ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_843c795c904b4dd89d0f9575184e65d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dee55037f8b3992732fd8f8e86333110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4930560290813446], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5ac4c2a16e8d334111fbbef89623186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_12caeb6ede33c8f15ccad23c01b02550(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82017cc3daeabbb57c90f49ab2be5f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac0ae25d3f54746d7b9bb8b633be03e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9526aff86654ce4c264a0db005edfe69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd667a1c8b27e86a760d9efef6c29bc
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0b28ecaf5eba55bf43bd6acdc0426de7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4ffad22801e8f82df86f11804ec2d97b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c48cab54675260e2404b6155ca59c569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ffad22801e8f82df86f11804ec2d97b
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4a68984cb0e6e4cf6dc1e6034b3c7d3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac0ae25d3f54746d7b9bb8b633be03e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93cdd17b8d75123febc16a987a9b70ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94f04d32a17310775c93c2bc96774a21
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4654c977cd69502d8f6bba4e89c98eac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89f85757158a2321ad582f23efd29c83
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e32acf4848d76270b5196a42767e0b5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb1289aa2dc645f78ee36a13f2917a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_09fca8ab22dc82aa9a2484c9d70da3de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.462415486574173], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9d0bc287f5e793ef0da4b6101fd0244d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25580fa142b192c65f743c2d3584165c
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_9c52e8b96d8987afca3b38329a1eab57(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_29d476b44dca6e56145ebbf3b11e2636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c52e8b96d8987afca3b38329a1eab57
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7343338131904602], dtype='float32').reshape([1]),
            paddle.to_tensor([0.27233320474624634], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b718a73be16451aa183e777e42f228e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c52e8b96d8987afca3b38329a1eab57
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7455248832702637], dtype='float32').reshape([1]),
            paddle.to_tensor([0.12875445187091827], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_9214128e22d67287a9b1338cc3fefba8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 1, 49, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 32, 16, 49, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_661ff19aa5ceb0445388739064e52f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9214128e22d67287a9b1338cc3fefba8
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_fd2cd28a80a2b191be308d82536c909a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_42f148f8c206425092b0bb8103d4a44b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2cd28a80a2b191be308d82536c909a
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_674db8cde5e3d13a57323930ac2aec05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f1d693c63ced0f99241259e4f9101c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac0ae25d3f54746d7b9bb8b633be03e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efd352d455c80cdbea102a98433bcbbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bbfcb13635d0ee466e6ef38a6192283
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 168, 168], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6e1f354f797fbd57a3b601570a3f870c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_685a056c59a4a7e4a8175935fc59f2f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 36, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a5b2203726ecb01558d5b9a91a11442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685a056c59a4a7e4a8175935fc59f2f0
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_deb9f3019b3527f328bececd239606a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6a656541f0f435fab771c6ed7e41840
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.01483928319066763], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_af833cfa6e363039d7484e2348ab7942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d669ea577ec945b43ddf91c31d43317(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3513548970222473], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c27fdc4770d3caa0669b5ced54c2e84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_523718ed6b8a648363b64ac42c3590ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91da848df7d9cb3e62ee12ef1594efa6
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f1356a59ce1751516ae8fe47d47dc90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2cd28a80a2b191be308d82536c909a
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47f7cabea8e4fcdf811e25cd38302a33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0ec7841af734e10357125abf937cfde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.019291505217552185, -0.27365997433662415, -0.3934049606323242, -0.16711992025375366, -0.018702328205108643, 0.26337409019470215], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0318598747253418, -0.007987573742866516, -0.21601513028144836, -0.025963157415390015, -0.06260842084884644, -0.07662948966026306], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7994c1a5a425fe9123e18f5d5c5db98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0006146249361336231, 0.002185879275202751, 0.08498142659664154, 0.004338960628956556, 0.0011709232348948717, -0.02018222212791443], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180fa63e8b15e1b08d462944a2b169f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0006146249361336231, 0.0, 0.0, 0.0, 0.0, -0.02018222212791443], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_438c0a62e26f71d46ffc604154ff2ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1580319106578827, 0.08440583944320679, 0.15580785274505615, 0.0, 0.0, 0.2714036703109741], dtype='float32').reshape([6]),
            paddle.to_tensor([0.24304825067520142, 0.0, 0.0, 0.16679298877716064, 0.09022623300552368, 0.0], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c96fedd38724aef6f91816dd93f2b2d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.019291505217552185, 0.04946009814739227, -0.3934049606323242, 0.016018692404031754, 0.011264503002166748, 0.3699343204498291], dtype='float32').reshape([6]),
            paddle.to_tensor([0.053903549909591675, 0.0039884597063064575, -0.061460524797439575, 0.1297207921743393, 0.06059029698371887, 0.05703727900981903], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_918326b4f86ee7089d4db17a32044c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.05832721292972565, 0.3405929505825043, -0.03126943111419678, 0.10864485800266266, -0.1369187980890274, -0.05729490518569946], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.05832721292972565, 0.3405929505825043, -0.03126943111419678, 0.10864485800266266, -0.1369187980890274, -0.05729490518569946], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f8b05c290b955360158be0a9e9370ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11661601066589355, -0.04659134894609451, 0.11697685718536377, 0.17422005534172058, 0.1380166858434677, -0.004758194088935852], dtype='float32').reshape([6]),
            paddle.to_tensor([0.11661601066589355, -0.04659134894609451, 0.11697685718536377, 0.17422005534172058, 0.1380166858434677, -0.004758194088935852], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f56624ab23fcf4eaabc6f362e7992b27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1580319106578827, 0.4075259268283844, 0.15580785274505615, 0.18313860893249512, 0.02996683120727539, 0.3779639005661011], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1580319106578827, 0.4075259268283844, 0.15580785274505615, 0.18313860893249512, 0.02996683120727539, 0.3779639005661011], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c9c80a8e9097db8b770b24399ed4763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2650919258594513, 0.011976033449172974, 0.1545546054840088, 0.32247692346572876, 0.213424950838089, 0.1336667686700821], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2650919258594513, 0.011976033449172974, 0.1545546054840088, 0.32247692346572876, 0.213424950838089, 0.1336667686700821], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_11ca1189d237ab37c2ecf4cb4d29728c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.09436476975679398, 0.951033890247345, 0.9186823964118958, 0.32256653904914856, 0.6005591154098511, 0.9353411793708801], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.232835590839386, 2.3465805053710938, 2.266756534576416, 0.7959005236625671, 1.4818192720413208, 2.3078603744506836], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_06f03019233eda98593c23a034fff7db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.02183729223906994, 0.6905632019042969, 0.6755805611610413, 0.2042846828699112, 0.47087711095809937, 0.683407723903656], dtype='float32').reshape([6]),
            paddle.to_tensor([0.021971477195620537, 2.231677532196045, 2.0824294090270996, 0.2567308843135834, 0.8899200558662415, 2.1586368083953857], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3b99904509c06aff9bbd9e5cbd9123a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76cab07479f76fa0c11ecb458cdbfe8
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1c731cd7d8d053e2b7043dd8fbcc08b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15adfb1bc330804d561de35ffad6122d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_56559bc3afe4ff3e2d411942dcbbe447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2cd28a80a2b191be308d82536c909a
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8d65d69ef173e84947f5d2745c2bc495(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8d65d69ef173e84947f5d2745c2bc495(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8d65d69ef173e84947f5d2745c2bc495(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8d65d69ef173e84947f5d2745c2bc495(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8d65d69ef173e84947f5d2745c2bc495(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffae275137ed45d2c00bba2cca0161f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffae275137ed45d2c00bba2cca0161f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8d65d69ef173e84947f5d2745c2bc495(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cd65f15b9b3693502cb70f9ca87815d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e97580dd9870b7e7b6fa7c8ce43e0072(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a4fdf50bec57d1409ecec3c91365c15b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97580dd9870b7e7b6fa7c8ce43e0072
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_22f38ff0bbfa0aa0496d94cd1df54a75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97580dd9870b7e7b6fa7c8ce43e0072
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e87c73a4085db765db31f50180dd1030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd667a1c8b27e86a760d9efef6c29bc
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7f1a1ad7772d4e7eadfe08b75f184117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_39292dd50fa0d8f3448f27b12049a7ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76fec3bf88c0e7b9b52efe4c6677bfbc
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dfcc9d193bc9c81e571d3a5afe84add5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_692ee6857b5f94d9df6cb98f233d9c59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.019055353477597237], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_011d9b155dcfbf64c3abe70fd5dfcd4e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_088d198c7fe0a136956dbcabc96b9e05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_011d9b155dcfbf64c3abe70fd5dfcd4e
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd339289402e7dfe5000be4bc5f91212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_843c795c904b4dd89d0f9575184e65d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e2ef6e89f369daba4459c22602052599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91da848df7d9cb3e62ee12ef1594efa6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_35a51197cbd4cd97360feaabd1e8a858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13c3663f29586eae9593bac13364d010
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5085993ce01c090896484eef91ecd0f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ad48c54adf7e753113a4039ad9c4b79
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1ef09689825f92f399eaa37195a08887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd667a1c8b27e86a760d9efef6c29bc
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba84c3ec59052f25d86ff9b510dd1578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_0e1256450ca42abefa9f4d8ecd58039e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cffc8540df118239ad6c137436b09879(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e1256450ca42abefa9f4d8ecd58039e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1cc69c49db65b38d79d59178420da99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7de945936a3451cd03b98ee1a6fe4502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2437657117843628], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_4639b3eb8f13649109dd5afc4e0a1c09(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 1, 49, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 8, 16, 49, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ca2971c2d0ad1dbd8ebcbb6c1bcb62cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4639b3eb8f13649109dd5afc4e0a1c09
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_af0da6db58e08cbd0cbfd11c5b5bd61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1521999835968018, 2.1796727180480957, 2.1047537326812744, 2.159518241882324, 1.822599172592163, 1.9034342765808105, 2.1630988121032715, 1.9198029041290283, 2.1094632148742676, 2.0728089809417725, 1.9138891696929932, 1.9151036739349365, 1.9168546199798584, 1.9518407583236694, 1.93830144405365, 2.3321938514709473, 2.16713547706604, 2.033275842666626, 2.194241523742676, 2.034022092819214, 2.0413191318511963, 2.2838566303253174, 2.2300868034362793, 1.9731148481369019], dtype='float32').reshape([24]),
            paddle.to_tensor([0.8327269554138184, 0.7016278505325317, 0.8010571599006653, 0.5192842483520508, 0.6323118805885315, 0.5801719427108765, 0.8297624588012695, 0.833179235458374, 0.5878900289535522, 0.7204785346984863, 0.9652197360992432, 0.748534083366394, 0.6652483344078064, 0.9230014681816101, 0.8385258316993713, 0.8157482743263245, 0.8609439730644226, 0.8209357857704163, 0.8064541816711426, 0.7680544853210449, 0.8674399256706238, 0.9423626661300659, 0.8580211997032166, 0.7216925621032715], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4a174c2b5f098db626f3b749b09502b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.222297430038452, 2.2627525329589844, 1.849575400352478, 2.053041696548462, 2.192687511444092, 2.159512519836426, 2.076664686203003, 1.9242665767669678, 2.056380271911621, 2.302868604660034, 2.204963207244873, 2.3179593086242676, 2.261704444885254, 1.8650039434432983, 2.3297932147979736, 1.9024587869644165, 2.0811641216278076, 1.9169301986694336, 2.161933422088623, 2.0917773246765137, 2.026906728744507, 2.07029128074646, 2.2546253204345703, 2.260570526123047], dtype='float32').reshape([24]),
            paddle.to_tensor([0.16727307438850403, 0.29837217926979065, 0.19894284009933472, 0.48071572184562683, 0.3676881194114685, 0.4198280870914459, 0.17023754119873047, 0.16682079434394836, 0.41211000084877014, 0.27952146530151367, 0.03478027507662773, 0.25146594643592834, 0.3347516655921936, 0.07699854671955109, 0.16147416830062866, 0.18425174057483673, 0.1390560418367386, 0.17906422913074493, 0.1935458481311798, 0.2319454848766327, 0.13256007432937622, 0.057637352496385574, 0.14197881519794464, 0.27830740809440613], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67722abb5a8d04b57aca373bbcd1f557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5409813523292542, 0.5511153936386108, 0.5134969353675842, 0.5270832777023315, 0.4896690845489502, 0.5027357935905457, 0.5370961427688599, 0.4801368713378906, 0.5218967795372009, 0.5342788696289062, 0.4810032248497009, 0.5041020512580872, 0.5080734491348267, 0.4862886369228363, 0.5003793239593506, 0.56325364112854, 0.5387951731681824, 0.5031106472015381, 0.5469971299171448, 0.5118545293807983, 0.5098521709442139, 0.5678868293762207, 0.5583927035331726, 0.5132789611816406], dtype='float32').reshape([24]),
            paddle.to_tensor([0.09662725776433945, 0.2517531216144562, 0.3178901970386505, 0.2152634561061859, 0.0554405152797699, 0.45703259110450745, 0.03614778444170952, 0.01393285021185875, 0.002307600574567914, 0.07740283012390137, 0.11828718334436417, 0.4843491315841675, 0.24832983314990997, 0.2953934073448181, 0.3565485179424286, 0.32441121339797974, 0.2690918445587158, 0.3854125738143921, 0.3194684684276581, 0.002826559590175748, 0.13202808797359467, 0.41044580936431885, 0.3000727593898773, 0.32590940594673157], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a8a0e6b028782608b81c44e0b229c515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00f23910479f0f2ee0a1f0a25caaf8bf
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9687598f80eb1b3665c885e1eb1ff5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d61564a854d48b3c32d0bba1c1804d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4c8b19199ddf4f084ed43cb3e70f61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_674cf7ec3cc9630b256f7a9c2538487b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_674cf7ec3cc9630b256f7a9c2538487b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_674cf7ec3cc9630b256f7a9c2538487b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_674cf7ec3cc9630b256f7a9c2538487b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7af2d4174f5334475826f5d31b62840d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7af2d4174f5334475826f5d31b62840d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7af2d4174f5334475826f5d31b62840d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dad506799b8a4bc031327aa36802d4e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dad506799b8a4bc031327aa36802d4e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dad506799b8a4bc031327aa36802d4e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dad506799b8a4bc031327aa36802d4e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dad506799b8a4bc031327aa36802d4e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63cde6cf4ef98a2d010b6351b23552e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63cde6cf4ef98a2d010b6351b23552e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dad506799b8a4bc031327aa36802d4e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1f420eb2d4ecee2e859708f8d6f3cf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1f420eb2d4ecee2e859708f8d6f3cf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d20cfd5ee21d926cac375053ff89e8c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd9cbd8acbc7f845c7de45667aa7f49e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24376803636550903], [0.24617058038711548]]], dtype='float32').reshape([1, 2, 1]),
        ]



class PrimitiveOp_aad4de51cbc89686b52f5e8b917a88d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3549, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8597a2ed7f75f3de70167334ff5047fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aad4de51cbc89686b52f5e8b917a88d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_a388d1fbfb3f180d2735b4decfef4415(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 16, 1, 49, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 16, 16, 49, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e8904ec0e966c9a0ff596246c44a099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a388d1fbfb3f180d2735b4decfef4415
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_bf653908cd5b8965bac4138fb7b97768(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_909bd79f86e28f7298149c0e3f240cc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf653908cd5b8965bac4138fb7b97768
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_982337694735f10f0b236a2b970ec976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_982337694735f10f0b236a2b970ec976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_982337694735f10f0b236a2b970ec976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_982337694735f10f0b236a2b970ec976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_81ffbda08ba2c29ec6b05758df97f64d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89f85757158a2321ad582f23efd29c83
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 136, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_06db875ee11d3984be920c7018b371db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_843c795c904b4dd89d0f9575184e65d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_79d17d46f1ec2b2cf1beab665b820a2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ad48c54adf7e753113a4039ad9c4b79
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b8b196bba9fe886a77f49befc217eb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07983528077602386], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fb2aaad7ac8d2d3bb7640c5feb587d34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac0ae25d3f54746d7b9bb8b633be03e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c378e25433f96c3d9e14a8fd9e63d43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c7749614d1ea334f8039f540f2f1c957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1816418170928955, 1.8633242845535278, 2.1522037982940674, 2.2153337001800537], dtype='float32').reshape([4]),
            paddle.to_tensor([0.9042313694953918, 0.7810401320457458, 0.5276364088058472, 0.7198778390884399], dtype='float32').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60b60e4909c9546299d2e4d6519e919e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.207533359527588, 1.9045121669769287, 2.0253593921661377, 2.0253467559814453], dtype='float32').reshape([4]),
            paddle.to_tensor([0.09576865285634995, 0.21895988285541534, 0.47236356139183044, 0.28012216091156006], dtype='float32').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_51a1dd579e9fedb844732b9619c61478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5460303425788879, 0.46808570623397827, 0.5230717658996582, 0.5405285358428955], dtype='float32').reshape([4]),
            paddle.to_tensor([0.14503681659698486, 0.010800845921039581, 0.08474092930555344, 0.08312032371759415], dtype='float32').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe916c13b8d42e302b3d080a2cab9f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe916c13b8d42e302b3d080a2cab9f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe916c13b8d42e302b3d080a2cab9f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe916c13b8d42e302b3d080a2cab9f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c64c6a57f3aa629a6e3809977e881d0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a70976acb84d41295856a0b44477c400
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d2aeac43f2ea4466d6ea853984f43a18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f10fa7dc2fe5284faae4eb27760576e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f47f8cde93bde2463ad8c9966f0858bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_92ec5b779a3fce6b93385d138336f9d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25580fa142b192c65f743c2d3584165c
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47f7cabea8e4fcdf811e25cd38302a33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_23eae88cc71ee7eb25bdbbc7f05dc137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685a056c59a4a7e4a8175935fc59f2f0
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_7f49ff7b6756f9be71426efbd30883ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_898ae09a3034f97b48eefdeae8b15ffd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f49ff7b6756f9be71426efbd30883ab
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3532690405845642], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1eeb9b6f0f20fabb3ca4bba981018876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0b15455d789d1cd911aaea6c15def69c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04022175073623657]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.32357287406921387]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc026026aaf0ae9b5c0e9b201d678261(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.23648566007614136]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.35454875230789185]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f8837bef47dd4801240fa286ad31b849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04022175073623657]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.30788999795913696]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e8904ec0e966c9a0ff596246c44a099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a388d1fbfb3f180d2735b4decfef4415
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a31f444d47524d01d518b2a8e6e936d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.02013292908668518], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.004770040512084961], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3829e1225eba948116986ba98bdc3316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.055064812302589417], [0.28067803382873535], [0.11762891709804535], [0.07574111223220825], [-0.1188349723815918], [-0.013302385807037354]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.07976526021957397], [-0.14685890078544617], [-0.15858013927936554], [-0.36050552129745483], [-0.013743840157985687], [-0.022245600819587708]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ca69ad638faf951042b3c664f830f38e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.16270190477371216], [-0.28220218420028687], [0.02013292908668518], [0.10196256637573242], [-0.014692038297653198], [0.24281780421733856]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3141138553619385], [0.06970024108886719], [-0.09054850041866302], [-0.2545846402645111], [0.10437272489070892], [-0.15421158075332642]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bdd5993025176145169ac6173c86af05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0719030350446701], [0.28067803382873535], [0.11762891709804535], [0.3329775035381317], [0.07311412692070007], [0.2764424681663513]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3891090750694275], [0.06970024108886719], [0.013089627027511597], [-0.23143820464611053], [0.1823386698961258], [0.07761061191558838]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a35bdc4c88429f4f87ae4a2917b63786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ead457af94b0b1f2965d42049c2051a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac0ae25d3f54746d7b9bb8b633be03e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4a9334425da00b42c6a7448b0a9a53b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2cd28a80a2b191be308d82536c909a
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_06b7b84fffb0fa45c7ecea3475e6d389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e7615c69dd71d274dc223ad336df9c9
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ee77eb569cc2a119eefcf8dbd92ab8de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89f85757158a2321ad582f23efd29c83
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a62b4593cd28bf9d0a9f2c27a4cad79c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 184], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_01006b683168baeed4e50d2471d7e24b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bf5744d1617a346b22175977c14d76a
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.6030163764953613]], [[0.7961981296539307]], [[0.6438075304031372]], [[0.7476810812950134]], [[0.8281542062759399]], [[0.754705548286438]], [[0.7412281036376953]], [[0.8092681169509888]], [[0.6864055395126343]], [[0.7921578288078308]], [[0.8171217441558838]], [[0.7205018997192383]], [[0.8045275807380676]], [[0.685869038105011]], [[0.7133346796035767]], [[0.8333665728569031]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2442f74887d56b1c3136b1872d71e5f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b8586a5f3cc41f21df895a760079769f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ae32797d9ee904be963c978a39a61d9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25a87951109d1a21d9596a75a3e62cba
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9d1b56be646767976a9091ec9720da96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00f23910479f0f2ee0a1f0a25caaf8bf
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ee0ca01110d62c0ad4c60eaf584c118e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6a656541f0f435fab771c6ed7e41840
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.29216843843460083], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_59f6eb32e499fc83683f299836630324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94f04d32a17310775c93c2bc96774a21
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_1368594ddabdc07823e4ef25b982b9d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d553fcfe812ebe89dadd912c422dc4a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1368594ddabdc07823e4ef25b982b9d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e8434fa071132293306a8ce176445ab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e8434fa071132293306a8ce176445ab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61ecc4470a0aa0d55cf10f51d2eeb104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd9cbd8acbc7f845c7de45667aa7f49e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24834126234054565]]], dtype='float32').reshape([1, 1, 1]),
        ]



class PrimitiveOp_eed90c1740ae68c7496dd62b2a738ba1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4116, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8114f8c6671085ffd69e4700d9a021f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eed90c1740ae68c7496dd62b2a738ba1
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_75802615ecd3acde3a224d79f6a71176(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25580fa142b192c65f743c2d3584165c
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_df52dfc2839d45eb69ce851818a9dfa5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 400, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 400, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7faddac00f222016f6f2575fce2b014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df52dfc2839d45eb69ce851818a9dfa5
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c18a78e0b683659f230576e6a3e9f835(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd667a1c8b27e86a760d9efef6c29bc
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a4bc3529cb45a286eb729f9b785366c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f0137164948409eb36e0253cae57891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13b1588f6e9741223c08664a57d8df13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13b1588f6e9741223c08664a57d8df13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13b1588f6e9741223c08664a57d8df13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13b1588f6e9741223c08664a57d8df13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c219d92876f21c705f75fcdb36123642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_934e944ba9c81a577462dce5e614d0d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1ef4b51e58b9c4e56b62b9017e4df78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5822a73324d862730f348f6bc267629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_31e0a488c61b713608eac8ce29327597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25580fa142b192c65f743c2d3584165c
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_84385b16b5c91c6b856325b2147cb0ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eb31c02dc2d0721f80c7d6802e16c847(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da89849831834739a5f4bc6bcd0ec3b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_bb7533bfe2ab31d32437f21c72cd99f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 336, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_74e1d6cce599a7c29a07c45e50d69eb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb7533bfe2ab31d32437f21c72cd99f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a627cdc6584fe4ee00d3df90fe9c6dec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e2d7ff5ed8ab963dd818f0d405e1a80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25580fa142b192c65f743c2d3584165c
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a9a27b0783305d60c0dcda84c1fc261(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_011d9b155dcfbf64c3abe70fd5dfcd4e
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ca2971c2d0ad1dbd8ebcbb6c1bcb62cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4639b3eb8f13649109dd5afc4e0a1c09
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d3c905c28367427b8fd50ae49d8768e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2cd28a80a2b191be308d82536c909a
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5beed8fc05e99e0731baf9559de4f40d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89f85757158a2321ad582f23efd29c83
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f60c667b0241e05e213057e568f47a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f60c667b0241e05e213057e568f47a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f60c667b0241e05e213057e568f47a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5124383e41eb51120b1f0a9a968fd1bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da89849831834739a5f4bc6bcd0ec3b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99364a3da8e3bf31c3babd5916e0cc10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_254f22656f03131bb87e84457a937586
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bd8dc389fc3aaa96bdb97d5a373ef022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_84e3ccef2d2457c5389feb3cfb776062(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df52dfc2839d45eb69ce851818a9dfa5
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b8f4c7d6c5d2a3dd22fd3e860aa7765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14183548092842102], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6a363a78addf4ea6b1503f8f84dc226a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ee0f8b495323f939eb9b18a3ef1083b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_709308be28153ee5be9193e1f457128e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee0f8b495323f939eb9b18a3ef1083b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_5737964e41058fb47e14b104c0f726a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 56, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 56, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f3a789d43a56aa81f1457b931bab35c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5737964e41058fb47e14b104c0f726a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9d3a621ada0bfa8159e9095aa093995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dbb5713fb4b7bae5ce361f4225462332(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dbb5713fb4b7bae5ce361f4225462332(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dbb5713fb4b7bae5ce361f4225462332(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dbb5713fb4b7bae5ce361f4225462332(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dbb5713fb4b7bae5ce361f4225462332(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c47f5a62ece50c4d4c09c1c0de29e948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c47f5a62ece50c4d4c09c1c0de29e948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dbb5713fb4b7bae5ce361f4225462332(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d99cae1149769eac2d652f817090b895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_843c795c904b4dd89d0f9575184e65d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6ce5c724bdd0ec4e53cf2edfe7ba846a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c52e8b96d8987afca3b38329a1eab57
    def get_inputs(self):
        return [
            paddle.to_tensor([0.9832037091255188], dtype='float32').reshape([1]),
            paddle.to_tensor([0.09283372014760971], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9dbab35e0c75293dc1bf852fd243ff1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c52e8b96d8987afca3b38329a1eab57
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8160135746002197], dtype='float32').reshape([1]),
            paddle.to_tensor([0.10435719788074493], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_be991e6edc87be246723671f0b10a65c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c52e8b96d8987afca3b38329a1eab57
    def get_inputs(self):
        return [
            paddle.to_tensor([0.854397177696228], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4671162962913513], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3a97bdbb565da34b4c9ab72f421b8cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c52e8b96d8987afca3b38329a1eab57
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8769496083259583], dtype='float32').reshape([1]),
            paddle.to_tensor([0.19629746675491333], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_71b6635479195413538a277cd76f459b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c52e8b96d8987afca3b38329a1eab57
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8831506371498108], dtype='float32').reshape([1]),
            paddle.to_tensor([0.30381250381469727], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ff8be8fc4c893655292e37aabb61ac1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c52e8b96d8987afca3b38329a1eab57
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8312440514564514], dtype='float32').reshape([1]),
            paddle.to_tensor([0.18074680864810944], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4e669a18c640b158a5d212e7246b5973(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c52e8b96d8987afca3b38329a1eab57
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7418454885482788], dtype='float32').reshape([1]),
            paddle.to_tensor([0.05117175728082657], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_129044cd3a0e93027656113bea1d3b10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c52e8b96d8987afca3b38329a1eab57
    def get_inputs(self):
        return [
            paddle.to_tensor([0.9902271628379822], dtype='float32').reshape([1]),
            paddle.to_tensor([0.05169527232646942], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1bb82d6e15fe461aa8d921cecff04fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c52e8b96d8987afca3b38329a1eab57
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8396027684211731], dtype='float32').reshape([1]),
            paddle.to_tensor([0.3299325704574585], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc1e88d4164e0432a0e749657cfdb039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25580fa142b192c65f743c2d3584165c
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68019eed6e7afdef9f3ca6eaa411efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68019eed6e7afdef9f3ca6eaa411efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68019eed6e7afdef9f3ca6eaa411efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68019eed6e7afdef9f3ca6eaa411efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4abd367406fc8384324b97f940d48fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_90a6b07624810a3be5f58542d7e61460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_53286e146f83d9a7e1c1af61ee6fbc54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2cd28a80a2b191be308d82536c909a
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eefb86d3caa6530a3afbf059dafff411(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed22ccee65afb4350547a5771b0998e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.40025511384010315], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eefb86d3caa6530a3afbf059dafff411(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed22ccee65afb4350547a5771b0998e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.40025511384010315], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b80c2a03b49bed2b0aed404a0bbe6dca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07e1134c7135d09d326f33d240f511c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bafa9df3fca11d878bfa520aef2d8f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ae5bb312abb4b2365e3e5e783f4a174
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.32611703872680664], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fb07917d7c10e45cbb3e548079b840e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a70976acb84d41295856a0b44477c400
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d5fcb9074324e1e1e33f459a266ce6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da89849831834739a5f4bc6bcd0ec3b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80ecb76ad4b21bd49e277c75596bec92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80ecb76ad4b21bd49e277c75596bec92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80ecb76ad4b21bd49e277c75596bec92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80ecb76ad4b21bd49e277c75596bec92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80ecb76ad4b21bd49e277c75596bec92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ff7a7e7905b60028cccd5d9c38c376af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ff7a7e7905b60028cccd5d9c38c376af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80ecb76ad4b21bd49e277c75596bec92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_42f148f8c206425092b0bb8103d4a44b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2cd28a80a2b191be308d82536c909a
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_30c50d3ce7b58ce35cd924675b1afe78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_208c3d9efa2e46e9c4051c56489fee0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30c50d3ce7b58ce35cd924675b1afe78
    def get_inputs(self):
        return [
            paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.21986964344978333]], [[0.2373102456331253]], [[0.24102632701396942]], [[0.2785630226135254]], [[0.47145286202430725]], [[0.2085183560848236]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6e2b6536dc1326b25769cc96a05855e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_843c795c904b4dd89d0f9575184e65d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b1682f7770f4bfdac2cf47f0e0d18de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b1682f7770f4bfdac2cf47f0e0d18de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b1682f7770f4bfdac2cf47f0e0d18de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b1682f7770f4bfdac2cf47f0e0d18de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b1682f7770f4bfdac2cf47f0e0d18de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_baa19fcf19397602873ee625e5a3b763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_baa19fcf19397602873ee625e5a3b763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b1682f7770f4bfdac2cf47f0e0d18de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_09990a50912293f748745a46fa8200b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14d5f61246d6ea1a7743b7e03031aed2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf5339c75b5c18e1bd066f80fe2eea21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_faa3c4acbaaa09b4587bb97eb2f308bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67491636f91b315c0cec4b6051b1e24b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2274e90fda78d9b76c039bd5027e0f39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a93e3e405bb7d2f1f8b6f24c056b8f36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bbccbad16d16dc3eef4eeb38216ccefb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b886ef05c84a9fac5f927a23deb9fe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1a6174ea50dc482e89d3a4963ae2cdfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_39292dd50fa0d8f3448f27b12049a7ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76fec3bf88c0e7b9b52efe4c6677bfbc
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec977f4cd7fe339a99bc70adfb5243fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4a9334425da00b42c6a7448b0a9a53b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2cd28a80a2b191be308d82536c909a
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_037222e7b8c536807081ef6b98ed1414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8b4a70fa6d659f2837c5fcbfd4157b94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bbfcb13635d0ee466e6ef38a6192283
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29e04ac46758bf3ab7f17e1dbe67580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29e04ac46758bf3ab7f17e1dbe67580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29e04ac46758bf3ab7f17e1dbe67580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efa42a106b56fbc9d0c6a1fc52b7d01e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bbfcb13635d0ee466e6ef38a6192283
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fca340a03f0dc63414cdfb6c66e75b36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fca340a03f0dc63414cdfb6c66e75b36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a41ad84512f69f60132779eb67f77933(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2016078680753708], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4b47c1ed180b0eb1f1eabc8f0070f518(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bbfcb13635d0ee466e6ef38a6192283
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8eab8735b4cdf331ba7f8761e9afd2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8eab8735b4cdf331ba7f8761e9afd2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8eab8735b4cdf331ba7f8761e9afd2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8eab8735b4cdf331ba7f8761e9afd2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_94170ae1faef82ce6069dbc44c2ff7ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 200, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 200, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_14fd53a30d3885ebabc6b03dffe272c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94170ae1faef82ce6069dbc44c2ff7ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_741cfe6de820ef46c6c7d3ebce651373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_191d36ce3e5905aeb7d38242628e2653(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 1, 49, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 32, 16, 49, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d722141c43a4b8560b8af9206f1700b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_191d36ce3e5905aeb7d38242628e2653
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3af9b98add13c4007e40a4e087812c66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a10b9c0cb44b68461521ecd20c14a9f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3af9b98add13c4007e40a4e087812c66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a10b9c0cb44b68461521ecd20c14a9f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3af9b98add13c4007e40a4e087812c66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a10b9c0cb44b68461521ecd20c14a9f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_5f5711f309b4fe387029065fecb338f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_75e8d51079547baea68e823e15c58542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f5711f309b4fe387029065fecb338f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3ae5ae3a8fc6ff40a37d075cc4e7891b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_843c795c904b4dd89d0f9575184e65d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7e6bda576d5ec3ddbd2e5318dcdb1b5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7e6bda576d5ec3ddbd2e5318dcdb1b5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3f0fb22b14a8db65dac163eb3afc4be2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.11621835827827454], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.22290584444999695], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_117cb4bf73a976943a54a54fbfe5624e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.06990808248519897], [0.19953130185604095], [0.2397157996892929], [-0.06576010584831238], [-0.17019523680210114]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.34975314140319824], [-0.14332401752471924], [0.01684647798538208], [0.24763968586921692], [-0.11009906232357025]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c2eafe978907301132493289e81bd962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.19966775178909302], [-0.07468777894973755], [0.11621835827827454], [0.10971325635910034], [0.14329633116722107]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2398657500743866], [0.2639036178588867], [-0.2593464255332947], [-0.402464359998703], [0.1149199903011322]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8380d0565bfb8cd30c72791d13c3bb15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.06990808248519897], [0.30127468705177307], [0.2397157996892929], [0.3520749509334564], [0.14329633116722107]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.3667130470275879], [0.2639036178588867], [0.01684647798538208], [0.24763968586921692], [0.209987074136734]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f7cff92938c40766c5d56fd8967cac4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bbfcb13635d0ee466e6ef38a6192283
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b851227d0d18c42c4d9c8482149433c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89f85757158a2321ad582f23efd29c83
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3338d5a322ced346ee9ab4d0cf5a9c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89f85757158a2321ad582f23efd29c83
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_92ef1838f01dabe47d9d4d923bd2f52d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4541bd25202b745afa301d2ba210f189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_71b4f7ec8b6ce2643c720a06eabd1dbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bc7b8a20e75d07d97b4011e8b993cf54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71b4f7ec8b6ce2643c720a06eabd1dbc
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4c8b19199ddf4f084ed43cb3e70f61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d61564a854d48b3c32d0bba1c1804d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9687598f80eb1b3665c885e1eb1ff5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96b4d9789265aa21c2be90c5f2f9a9c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2afa241435484f1647a6e061f02e241f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int32').reshape([]),
            paddle.to_tensor(8, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_503c16d0a7f132a28679faefe35c08a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6a656541f0f435fab771c6ed7e41840
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.011590427719056606], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ad5104bb67547045f0c8d854626c60a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b21ff1ab3efead121219b2820c5d9d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67ad212ddf587faa72ca52c5f570fe35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_189fa35562a02b0e633b212bbfb7256a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf653908cd5b8965bac4138fb7b97768
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67d4942f6fa5de130fe140f4fb9a4038(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_843c795c904b4dd89d0f9575184e65d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1811be0712bbd47dbb4df37b80f75bc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13c3663f29586eae9593bac13364d010
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_14efff7fb8eab39a5a8c1ae61ed53435(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f49ff7b6756f9be71426efbd30883ab
    def get_inputs(self):
        return [
            paddle.to_tensor([0.27201327681541443], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5cff3f6b66919926b8a5b9d3f3f6ea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a10b9c0cb44b68461521ecd20c14a9f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5cff3f6b66919926b8a5b9d3f3f6ea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a10b9c0cb44b68461521ecd20c14a9f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5cff3f6b66919926b8a5b9d3f3f6ea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a10b9c0cb44b68461521ecd20c14a9f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2e7c528bb4e4a74bdc83aaec5e536e78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f5711f309b4fe387029065fecb338f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_1b7e30b8bc0eeb49e535db2f3424d4a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1248, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fef29352f9087106125da6cde0f46938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b7e30b8bc0eeb49e535db2f3424d4a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f0666cec2be17f97ac6031ae32c5f334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf653908cd5b8965bac4138fb7b97768
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_393dfa4142a63af91ebf31cb56d15ede(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed22ccee65afb4350547a5771b0998e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.05977138504385948], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_393dfa4142a63af91ebf31cb56d15ede(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed22ccee65afb4350547a5771b0998e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.05977138504385948], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d799e93e80bafa0710f5ea3e6c58f600(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07e1134c7135d09d326f33d240f511c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_53c549de563407467ec389b4fad6f8af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ae5bb312abb4b2365e3e5e783f4a174
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4519156515598297], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e39a887b78cb2460a1376fc32b9060cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e39a887b78cb2460a1376fc32b9060cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e39a887b78cb2460a1376fc32b9060cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2099d5ebe8995443f595432ce664e7c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76cab07479f76fa0c11ecb458cdbfe8
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ede582b6acf55247c0f8cdc42da718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ede582b6acf55247c0f8cdc42da718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ede582b6acf55247c0f8cdc42da718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebe43fa5da4b58112ca3e2dfb4c3834d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebe43fa5da4b58112ca3e2dfb4c3834d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebe43fa5da4b58112ca3e2dfb4c3834d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e7983b2a781f88dee694c356ef39f178(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685a056c59a4a7e4a8175935fc59f2f0
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9d9338a5d958beebfefbad8d2f56dc36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71b4f7ec8b6ce2643c720a06eabd1dbc
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1010690be57618adbb907184fad52c07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2758355438709259], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4799cd095b5f004e4fcd23b176f1e1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89f85757158a2321ad582f23efd29c83
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_54f3b88a242f84350db8f87656d3a744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac0ae25d3f54746d7b9bb8b633be03e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4568a46d16fcb92dab5e8cec54b67661(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13c3663f29586eae9593bac13364d010
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63b0258a1fc2d9e0ba9689b8e3fe0826(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63b0258a1fc2d9e0ba9689b8e3fe0826(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63b0258a1fc2d9e0ba9689b8e3fe0826(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63b0258a1fc2d9e0ba9689b8e3fe0826(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63b0258a1fc2d9e0ba9689b8e3fe0826(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cd6059941e2acbacb0a09bf144d0b808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cd6059941e2acbacb0a09bf144d0b808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63b0258a1fc2d9e0ba9689b8e3fe0826(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_996669934570c068c4a14e82b4f3a5bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_90392ba2ca7d7df6f40f5cb31e2d98a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2685d6626ea55ece494efc935a7aac67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2685d6626ea55ece494efc935a7aac67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2685d6626ea55ece494efc935a7aac67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2685d6626ea55ece494efc935a7aac67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2685d6626ea55ece494efc935a7aac67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_69ca9f16e99ddcb52188bed1b25f43d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_69ca9f16e99ddcb52188bed1b25f43d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2685d6626ea55ece494efc935a7aac67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_50dbf80a642d2e0bed45ef079c63b9e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_50dbf80a642d2e0bed45ef079c63b9e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_50dbf80a642d2e0bed45ef079c63b9e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_50dbf80a642d2e0bed45ef079c63b9e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_50dbf80a642d2e0bed45ef079c63b9e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea824272582fdcba46349dc4f3c649ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea824272582fdcba46349dc4f3c649ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_50dbf80a642d2e0bed45ef079c63b9e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_585db02378269ac5cb1ec83cc7db87c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed22ccee65afb4350547a5771b0998e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2915319502353668], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_585db02378269ac5cb1ec83cc7db87c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed22ccee65afb4350547a5771b0998e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2915319502353668], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_284fa151bf34b1aa042141e66e6bec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07e1134c7135d09d326f33d240f511c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60af0a8b84d6648d3c58de171a76e49b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ae5bb312abb4b2365e3e5e783f4a174
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.11829502135515213], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_ec52547034c86e10ab00c5aef65fdfe3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 156, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9d7b87b3218a482c8ce86779e681bf23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec52547034c86e10ab00c5aef65fdfe3
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_226f7508aa262d90d10307d426c7e608(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_14d3bd62e54b9ff11a9a43f71cb29e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_226f7508aa262d90d10307d426c7e608
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.7349135875701904]], [[0.9064015746116638]], [[0.8643157482147217]], [[0.8141332864761353]], [[0.9224852919578552]], [[0.891898512840271]], [[0.9145889282226562]], [[0.8929387927055359]], [[0.7257941365242004]], [[0.8682162165641785]], [[0.8355649709701538]], [[0.8789184093475342]], [[0.8609952926635742]], [[0.841895341873169]], [[0.8884779214859009]], [[0.8728126883506775]], [[0.8682401180267334]], [[0.8504614233970642]], [[0.866704523563385]], [[0.7024329900741577]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]



class PrimitiveOp_44a1be71414800c97cf82ac1495ff298(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_caf950ee6c175cd7b35efb04981919f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44a1be71414800c97cf82ac1495ff298
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_be774afbd3bca632209dc341fb8a17b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 32, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f0956dff38c4adbd1755e95077cc79c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be774afbd3bca632209dc341fb8a17b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4f2f990e76aa1b4b04e6fe49ec41fbc1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 16, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15caa7f9cbd9e30756ea01f971512b35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f2f990e76aa1b4b04e6fe49ec41fbc1
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_719c0effb808cd3aa4ce37ed2ed24258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_113835ffad837d2f5aa6fe921e172709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_79ded6f53737680ce540c4ffe5ad4bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_357c94f4338f69e44035926707dae854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94170ae1faef82ce6069dbc44c2ff7ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebde44242cd7bb0df61832e528d3361a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_91e09b7bd9eaae2ce31a3547c7ec9772(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e7615c69dd71d274dc223ad336df9c9
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68f16f54db703f5768bdf72c4765fc83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ded4eb5408abf9cc641bb9821611450
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68019eed6e7afdef9f3ca6eaa411efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68019eed6e7afdef9f3ca6eaa411efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68019eed6e7afdef9f3ca6eaa411efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6121ddcae6e7ed594fa6dec7921b5fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f49ff7b6756f9be71426efbd30883ab
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20080240070819855], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f47cc94ff9c8192c9938295d59758ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_495d0b01d2131edffcf6b157509bce65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9488ab4ee3391ae79115b34d2f0d7d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eb35d82d327f739cda404403457797aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55debdc208bedcff501c1e3390963ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_843c795c904b4dd89d0f9575184e65d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0b3d74b7cc26e4642aabe1a6b646d205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b885a9beea045cc51cf748579924e835
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_de4deaea7dda2af08735523d167d8ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b7f566578acbb10c4aff64abccde67e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_00d7030bc89589bea0f121a6869a68e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 1, 49, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 8, 16, 49, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_54fd79bb9a096b217b80ec6bcc7b2a62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d7030bc89589bea0f121a6869a68e0
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f897d6015e6137cecdbdf7f6acb20c95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76cab07479f76fa0c11ecb458cdbfe8
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cdb100e1b7ec1c394cfcfcf5b1451442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c13876966c9e62bb2b8d3893967019ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3f928ef05d4c320363cb3b7a3ff8ec34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76cab07479f76fa0c11ecb458cdbfe8
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc2e74546d8cb5aab39861932c1c7a71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3866167664527893], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_9a075f5f8e8ca57465207b03d2cdc83c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e0196ccf0a2f02009c67f22cd74300ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a075f5f8e8ca57465207b03d2cdc83c
    def get_inputs(self):
        return [
            paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.37492507696151733]], [[0.31706711649894714]]], dtype='float32').reshape([2, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f335a7dc399a9fc60d0743db251f0c0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_843c795c904b4dd89d0f9575184e65d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4abb456a37be9b9ba3438d30b0749614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685a056c59a4a7e4a8175935fc59f2f0
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_9f1fa05d3a54c00a69211d5ef69c863d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 1, 49, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 16, 16, 49, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffc81f3bd1fb12f6f43386dd8dbb8829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f1fa05d3a54c00a69211d5ef69c863d
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d909de3384d7458de0f1d09e0457f52c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_18f73279f4a09b12f1d5b0b80bc3767a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e16b67e2edb439ecce1d5fd12c9b5a90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18f73279f4a09b12f1d5b0b80bc3767a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_19de1220e385d7e68a6e38e7880d71bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac0ae25d3f54746d7b9bb8b633be03e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_d0b268fec4575036f79efeb6bfa7a69c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_88141c7da36440e3fa9b81b914671596(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0b268fec4575036f79efeb6bfa7a69c
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7fbab6b6d40ae9c88dcebc5566bfb8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf653908cd5b8965bac4138fb7b97768
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_661ff19aa5ceb0445388739064e52f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9214128e22d67287a9b1338cc3fefba8
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32860c1486194a2fc5007877848db5ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97580dd9870b7e7b6fa7c8ce43e0072
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5cae51af79abf1158f1d4a3a89d18f36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1815907955169678, 2.2769126892089844, 1.907932996749878, 2.0796091556549072, 2.110363006591797, 2.1431312561035156, 1.9416979551315308, 2.245042562484741, 2.209714651107788, 1.8555011749267578, 1.881988525390625, 2.0004849433898926, 2.0412535667419434, 1.860338807106018, 2.17122745513916, 1.93278968334198, 2.2222325801849365, 2.042781114578247, 2.1074957847595215, 2.1499881744384766], dtype='float32').reshape([20]),
            paddle.to_tensor([0.5526050925254822, 0.9161155819892883, 0.8250937461853027, 0.7532532811164856, 0.8525898456573486, 0.8517947196960449, 0.5593414306640625, 0.6254949569702148, 0.9594743251800537, 0.5187383890151978, 0.5623292922973633, 0.7030147314071655, 0.7970205545425415, 0.8406480550765991, 0.7951136827468872, 0.7482725381851196, 0.6160892248153687, 0.6457232236862183, 0.6575725078582764, 0.507962703704834], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_657f4d0630d559dfc733b48252df9139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.0350992679595947, 2.1021230220794678, 2.1693103313446045, 2.033270835876465, 1.9085214138031006, 1.9629747867584229, 2.2445266246795654, 1.951501488685608, 1.934584379196167, 2.248936414718628, 2.3342783451080322, 1.8598324060440063, 2.115058183670044, 2.0682835578918457, 1.9403040409088135, 2.1461122035980225, 2.099975109100342, 1.9092484712600708, 2.2656540870666504, 2.1662161350250244], dtype='float32').reshape([20]),
            paddle.to_tensor([0.4473949074745178, 0.08388441801071167, 0.17490623891353607, 0.2467467337846756, 0.14741015434265137, 0.14820529520511627, 0.4406585693359375, 0.37450501322746277, 0.04052567854523659, 0.48126161098480225, 0.4376707375049591, 0.2969852685928345, 0.2029794454574585, 0.15935193002223969, 0.2048863172531128, 0.25172749161720276, 0.38391077518463135, 0.35427677631378174, 0.34242749214172363, 0.49203726649284363], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62764f929d6bf0f1a0018d2e4d404a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5290127992630005, 0.5655626058578491, 0.48841238021850586, 0.5170438289642334, 0.5201523900032043, 0.5291078090667725, 0.5187854766845703, 0.5337774753570557, 0.5496411919593811, 0.511211633682251, 0.5199856758117676, 0.4896783232688904, 0.5140585899353027, 0.4733687937259674, 0.5309785604476929, 0.4966222047805786, 0.5438241362571716, 0.4988684058189392, 0.5404133796691895, 0.539493203163147], dtype='float32').reshape([20]),
            paddle.to_tensor([0.06382325291633606, 0.4575190544128418, 0.42838215827941895, 0.3507518470287323, 0.2551310062408447, 0.3879965841770172, 0.2360391467809677, 0.20912696421146393, 0.2905004322528839, 0.49082446098327637, 0.013807285577058792, 0.32814526557922363, 0.09752397239208221, 0.26693081855773926, 0.08003351092338562, 0.252758264541626, 0.3814542591571808, 0.08651839196681976, 0.21119371056556702, 0.016675109043717384], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8bf691272b6627c9e0036b6ea705aa96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_242ef02814bc3e9b04a5e3785a627c64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6da6cb61414911df41d0c2a164248997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_242ef02814bc3e9b04a5e3785a627c64
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bd9b4b5fda13cb216e2b1af578007d05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc509b17341f2599b44cd69b7fa49182(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94f04d32a17310775c93c2bc96774a21
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_24340a8718a5fab9612db5f22dba74e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1368594ddabdc07823e4ef25b982b9d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cf38b48b32a1d2ab55a0ad86687656e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c73b705f6f5849fc980e28da2efc3ddb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bbfcb13635d0ee466e6ef38a6192283
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c9e1be4609210d775065a4eacbbee69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bc2118009851a836ce0d57314b82efb
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.8911274075508118]], [[0.8892913460731506]], [[0.9355054497718811]], [[0.8367183208465576]], [[0.83566814661026]], [[0.8696799874305725]], [[0.8848949670791626]], [[0.8737568259239197]], [[0.9251277446746826]], [[0.8608295321464539]], [[0.8690536618232727]], [[0.8794706463813782]], [[0.8996880650520325]], [[0.8204582333564758]], [[0.8965553045272827]], [[0.7918619513511658]], [[0.8368157148361206]], [[0.9273790121078491]], [[0.8898094892501831]], [[0.9457544088363647]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0519321d62aa40fa2cd6b0540ea465bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c13876966c9e62bb2b8d3893967019ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2387de030eec630fdcf01f6d035e27c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da89849831834739a5f4bc6bcd0ec3b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_495d0b01d2131edffcf6b157509bce65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9687598f80eb1b3665c885e1eb1ff5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d61564a854d48b3c32d0bba1c1804d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4c8b19199ddf4f084ed43cb3e70f61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9f3f811fec241d9a97a477286850061c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a70976acb84d41295856a0b44477c400
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8d3fc465ee6b8e783fa4da8b322cdb94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25580fa142b192c65f743c2d3584165c
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eb35d82d327f739cda404403457797aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c2532ff37733472ef6bb5123f4a4b436(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89f85757158a2321ad582f23efd29c83
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d3c905c28367427b8fd50ae49d8768e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2cd28a80a2b191be308d82536c909a
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7df2d4c80d5fdc405285d10f3e78b2f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f47f8cde93bde2463ad8c9966f0858bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_524e2ba8c13903f5228b5c4912a630ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15adfb1bc330804d561de35ffad6122d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_54fd79bb9a096b217b80ec6bcc7b2a62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d7030bc89589bea0f121a6869a68e0
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37792ab0a254f950e1001a0bbcabcc0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2cd28a80a2b191be308d82536c909a
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3769081db9813e3d0df871c846d5cc35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb7533bfe2ab31d32437f21c72cd99f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_126a8dd9f2cb244d8830084cb10e21f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2cd28a80a2b191be308d82536c909a
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3eba29e63b67380f45eba1e35bb2edd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_843c795c904b4dd89d0f9575184e65d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea4039a68c0586f3f5f358b739fa7e25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5737964e41058fb47e14b104c0f726a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9ed57930f63b0c5e95f08aa065f4087a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94f04d32a17310775c93c2bc96774a21
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fcef7fb987fa14a04ac9f7b6fdea98d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c532038966272a76e2f0a3f650179278
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e8672b972d7ac7fcad31678ce6d1f2af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.024495869874954224]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ddb5b947c2fff43b17b363d7e0f8f858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.339305579662323], [-0.08985428512096405], [-0.1387719362974167], [0.19930914044380188]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.051651835441589355], [0.1649070680141449], [-0.02027750015258789], [0.042431771755218506]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa204298ad4541231a82cf99a429f2e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09980668127536774], [0.26724952459335327], [-0.19716008007526398], [0.13058194518089294]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.10653990507125854], [-0.19285885989665985], [-0.06860601902008057], [0.12275975942611694]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c6b24196eedb95c9d36bc3ac2509b68d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09980668127536774], [0.26724952459335327], [0.03494563698768616], [0.3053952157497406]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.22915056347846985], [0.1649070680141449], [0.0039511919021606445], [0.18009227514266968]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7cf1247dcc6c967a8500e9e7334ff3c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da89849831834739a5f4bc6bcd0ec3b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_81adde871c8e823266831f3035f4f803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba84c3ec59052f25d86ff9b510dd1578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b78a681bbef3aa28027b0fc3258f381(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ded4eb5408abf9cc641bb9821611450
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_fe7fc4efabeedecabaace5c4f606eb7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_17f05154f8e876a59f3899df22c24c1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe7fc4efabeedecabaace5c4f606eb7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f3156bf6c60ed22c756d73deef25ce4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bbfcb13635d0ee466e6ef38a6192283
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c0d8ab61a5d640b66116bd29e11f9ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e7615c69dd71d274dc223ad336df9c9
    def get_inputs(self):
        return [
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e85aa88181dc9c4cd42fc0eb80b138f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_934e944ba9c81a577462dce5e614d0d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2640974f97ba0df1d5b8a65bd1f6755c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2e190fc705e8c736c386769e0e84e307(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a70976acb84d41295856a0b44477c400
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99b14c6a840daf2ffb476016234f049d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99b14c6a840daf2ffb476016234f049d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99b14c6a840daf2ffb476016234f049d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99b14c6a840daf2ffb476016234f049d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99b14c6a840daf2ffb476016234f049d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f6a87cbbaa252affc39d095ad34103f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f6a87cbbaa252affc39d095ad34103f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99b14c6a840daf2ffb476016234f049d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e9ecb9e72353a197577c44103d6ed851(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2cd28a80a2b191be308d82536c909a
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_380f44caa5da5d2e21bb63a82752370a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac0ae25d3f54746d7b9bb8b633be03e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f76f5b20fd47eeffb07f785faf9156d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_934e944ba9c81a577462dce5e614d0d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c7cb70d8f2035c133816c93404dc2628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a10b9c0cb44b68461521ecd20c14a9f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cfc72ecd24ed2f43c7f4949c9be17f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5b32694be5cfeb1829cbfe8d0c27f540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25580fa142b192c65f743c2d3584165c
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e2b16cfeddf72d3ef52857e6cbb16e2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_76df5d9c80f5fe1455b23bbfb28fefa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2b16cfeddf72d3ef52857e6cbb16e2f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_88fffc57ada75d2b61d1e505cde0fbc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f49ff7b6756f9be71426efbd30883ab
    def get_inputs(self):
        return [
            paddle.to_tensor([0.26912549138069153], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c9e0b47dff53e82de6648d02338df155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bbfcb13635d0ee466e6ef38a6192283
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_17a22aca88dd28bf5927e221647383f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 2, 1, 9, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 2, 16, 9, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5281f86c22afac387db2696b32d147f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17a22aca88dd28bf5927e221647383f6
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 1, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_59e85d9072ded26b00aa2527ea4b0c0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed22ccee65afb4350547a5771b0998e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3996615409851074], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_59e85d9072ded26b00aa2527ea4b0c0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed22ccee65afb4350547a5771b0998e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3996615409851074], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bdfa8244b87764f25aa50388761e8933(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07e1134c7135d09d326f33d240f511c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fb58e4d572a9f56ab479f016f2aeced7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ae5bb312abb4b2365e3e5e783f4a174
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.062410056591033936], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e6a7033eb943a0e640efa9b577cb58b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94170ae1faef82ce6069dbc44c2ff7ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493681bce4b55406cb0e15107b0cec5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df52dfc2839d45eb69ce851818a9dfa5
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f16dfe49f572fefc695f716e8237c44a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4a822dcd1953dbb9535355c07633c1a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a9a27b0783305d60c0dcda84c1fc261(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_011d9b155dcfbf64c3abe70fd5dfcd4e
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9c6072e5e1e49bfb7af15c7c264be50a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebe43fa5da4b58112ca3e2dfb4c3834d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebe43fa5da4b58112ca3e2dfb4c3834d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebe43fa5da4b58112ca3e2dfb4c3834d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebe43fa5da4b58112ca3e2dfb4c3834d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d722141c43a4b8560b8af9206f1700b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_191d36ce3e5905aeb7d38242628e2653
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5fb0f82fb91eda75405108cba1450fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b21ff1ab3efead121219b2820c5d9d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f77deb4272df7b4fedf149f58b4b8af6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c2bf3bcdf6c81e305a7f9185f1f5890(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89f85757158a2321ad582f23efd29c83
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_204cd1c5505f37ceb86b9db1c39d82b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_da4645cb8e139a7e55ada1140dbb6d2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204cd1c5505f37ceb86b9db1c39d82b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_af83c212de351e11b0d8b973ecaa6289(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_088d198c7fe0a136956dbcabc96b9e05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_011d9b155dcfbf64c3abe70fd5dfcd4e
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c942c2a837beb7f18cb07a5768554c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7f18e8fb07bf2f204bad5f0dd52d5aef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a70976acb84d41295856a0b44477c400
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aaaf7bacec4916c5197334ba8b60de17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ded4eb5408abf9cc641bb9821611450
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5cff3f6b66919926b8a5b9d3f3f6ea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a10b9c0cb44b68461521ecd20c14a9f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf4585bb352ed03f4b93b7aa2f3e50ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15adfb1bc330804d561de35ffad6122d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1c00cc521b79b55dfec3c3d567ba7fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6a656541f0f435fab771c6ed7e41840
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.41445696353912354], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a4bc3529cb45a286eb729f9b785366c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32e2e317924ab72d6f4f79c58e3948f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c349a69875b8814ad73a342307f0d562(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3598812520503998], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e6d824b6487013269aea85e1b7a9517d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a70976acb84d41295856a0b44477c400
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a4f62e8b0c85a49a6eb790e983282f39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e2ef6e89f369daba4459c22602052599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91da848df7d9cb3e62ee12ef1594efa6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ac1708840b1762075458bdedb9c5ef3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6a656541f0f435fab771c6ed7e41840
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4050699770450592], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_90392ba2ca7d7df6f40f5cb31e2d98a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_894923c7f8e14d861a4dbf99bd21d03c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31b48274dcd5e8f61211460521e2419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31b48274dcd5e8f61211460521e2419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31b48274dcd5e8f61211460521e2419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31b48274dcd5e8f61211460521e2419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_f52fe9f3958be47ff9be8d97992b8eb5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2a977ee00384f8da9f7a5a70e72f3b79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52fe9f3958be47ff9be8d97992b8eb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_320011ffd33ed2a79dac15636894acea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5611b87c49f8a839e58a228a2075a8bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c1f7ca8cd8a1011fea21c3b46525df1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9173893a0f093b8c9d90c05eec43ebb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a8f612a6834fa5c77df441c8d56c4a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_22617826637cd5b28edad1233865dc0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_39c3d42f1eb322a28ee9fc3e41df70f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c30799078384a3a2c582ebe067041d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80284c1fc37ea9e3e6a75904396378f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3248227536678314], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8733069dd5c8dc94e1386d718592a986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_934e944ba9c81a577462dce5e614d0d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9d8ddfb7300e7d300261d0a0e8845ab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf653908cd5b8965bac4138fb7b97768
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f1356a59ce1751516ae8fe47d47dc90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd2cd28a80a2b191be308d82536c909a
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_11535e13259917038414cf53e053dba0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.40715083479881287], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e03d310af881ce26f826dc47097e98a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4555abc0611a87943e03789d78ac7d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4555abc0611a87943e03789d78ac7d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4555abc0611a87943e03789d78ac7d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5772a7b307d1d8ea23166d676c58f9b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5772a7b307d1d8ea23166d676c58f9b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5772a7b307d1d8ea23166d676c58f9b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5772a7b307d1d8ea23166d676c58f9b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5772a7b307d1d8ea23166d676c58f9b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dc75e8102dbd2978555ee23a161088fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dc75e8102dbd2978555ee23a161088fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714d56a3218460f5481fdb9da264cf9c
    def get_inputs(self):
        return [
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5772a7b307d1d8ea23166d676c58f9b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f71451945d86e3ca9499287e41ed0ec
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9687598f80eb1b3665c885e1eb1ff5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d61564a854d48b3c32d0bba1c1804d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4c8b19199ddf4f084ed43cb3e70f61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ae31aa43af558c87e5e23d59259f19f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6a656541f0f435fab771c6ed7e41840
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4200989902019501], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_e90d7ceb91e546512b6928ef708cc4c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 624, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a4bbe2a92b43afa20203b40b82a31a1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e90d7ceb91e546512b6928ef708cc4c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b936accbab46a866d76fd1d58ffcfbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76fec3bf88c0e7b9b52efe4c6677bfbc
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4555abc0611a87943e03789d78ac7d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4555abc0611a87943e03789d78ac7d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4555abc0611a87943e03789d78ac7d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4555abc0611a87943e03789d78ac7d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f021507bf4c4821b1e224b30b78a0a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac0ae25d3f54746d7b9bb8b633be03e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffc81f3bd1fb12f6f43386dd8dbb8829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f1fa05d3a54c00a69211d5ef69c863d
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cedd00bf722df26b50e6bfa7d190af28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_08927a25e93982456857794e8e6da215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6a656541f0f435fab771c6ed7e41840
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31590908765792847], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b94ea20ce43805a3a55eca9b7764bfab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc52cae1dcd6cc3b092b9a6aa63253f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc52cae1dcd6cc3b092b9a6aa63253f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d934923499fdadc985e8bc2484b63af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_22642a05006096eb9728127b61f242ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2ae4f7665e8bbb1372fc89580d983511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1d5a0c7777888284e10094c429ce1f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 152, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_312960072daa56feca3a8c921546162a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b56af361903a87de4526784604cf1c4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[0.0]]], [[[0.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_08c911174fc6d1a0de4efdc175a93ba9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a754aa284ed7658aefb7a2d4db9dd56f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c3139c5b82e492bfbefd567659ac89f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f35c3615fb49592c50b7491726cb7c5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b48f9c97b5dcd576696f11f84ff6b659(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38c3c9561b3f583f86f8c6968f2e8feb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18265779316425323], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_45d12bcfcd740b2232b26dc425629c6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7007b580b4c1f6555bdd7ec43fb26a5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cff5ed252f0a59b0c497fed595c414b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cff5ed252f0a59b0c497fed595c414b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cff5ed252f0a59b0c497fed595c414b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0d8e115d5db0db99e5d98f97388b4448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9687598f80eb1b3665c885e1eb1ff5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d61564a854d48b3c32d0bba1c1804d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4c8b19199ddf4f084ed43cb3e70f61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0241c99fc31b377492ee91b4de5ebeea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.33804893493652344]], [[0.39942264556884766]], [[0.09662508219480515]], [[0.1508401781320572]]], dtype='float32').reshape([4, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e6055cd5a183c74c812596aae757825(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_20a5f4fb92b2cefa6029414695994d5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4b0569bac51415dd1bf6a7eac557d6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f02cd02a0c546707f0d6d78e29c5f12a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 1, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd32ef1590d03eb0fb741ec59089f7e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_310ee0450d2e1bf8e094d9c98137b3b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.40364521741867065], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe916c13b8d42e302b3d080a2cab9f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe916c13b8d42e302b3d080a2cab9f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe916c13b8d42e302b3d080a2cab9f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_927ada4cf9c337e80591e06923d0f699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.9297406077384949]], [[0.8804839849472046]], [[0.9003913998603821]], [[0.9182154536247253]], [[0.9530584812164307]], [[0.9100152850151062]], [[0.9088941812515259]], [[0.9383436441421509]], [[0.9679425358772278]], [[0.9257610440254211]], [[0.9264084100723267]], [[0.92100989818573]], [[0.9689187407493591]], [[0.8751189708709717]], [[0.9151067733764648]], [[0.8847223520278931]], [[0.9026293158531189]], [[0.9414359927177429]], [[0.8382887244224548]], [[0.9695025086402893]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c8372f058ee4b3e91b95a44749b4d74d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f10ea91e1b3716a84eb028fbd78900cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13d5fece80dafab772434771c4c8dfe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ab68a56f24e191815c488bbbb58aebd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc1117342399bb3c7b22e585664f0caa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc1117342399bb3c7b22e585664f0caa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_baacd7ce7cece1e299f1d7d04580b3e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24692094326019287]]], dtype='float32').reshape([1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c851cbcbb5d659873d63bf66e1dd093(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e9422dfe377e493927302b4471d0ce3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18271088600158691], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3f8ff48b7afabae7e9478bcac8ec66e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd1f95bb5e784c6d4308191662115631(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37c52ef01d90f745f96ca9d35e55840c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_322987b5d210c3cea18b5a5987382c5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f60c667b0241e05e213057e568f47a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f60c667b0241e05e213057e568f47a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f60c667b0241e05e213057e568f47a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f60c667b0241e05e213057e568f47a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c18c8ffcd95c88918fad1a09ee2d7288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_674cf7ec3cc9630b256f7a9c2538487b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_674cf7ec3cc9630b256f7a9c2538487b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_674cf7ec3cc9630b256f7a9c2538487b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cdf335e23b80419868d56bbefdaa198a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_733123a56e74442a69df6790486c7109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_de6751153a82bd4679988db4fe2acb2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b156014403960be5f44d36bb6e72cfa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d008ac142b4266ae3dd30014a0ae8488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc042faffb6f4479c797d66cabacc8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_51cb67d0bdecf87158abbd6efd3c4177(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c6018e10579c0d51ae56fd5af68e9db5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.11957564949989319], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a974ca95fcb96c28056439a758bb4609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_48fcd86a04c2d7749314dbd996e5f8db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d324f5fe9129e0f65aac67bf801200fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0ba7e8b04f21d5ca8d9473c991c176b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1ea4db3611945c4e2aa9085f008213f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.284286230802536], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f49ee4db72790a62928d46a60dda9764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4774e2d8392a1d6d14192c5baaa1600f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.761617124080658]], [[0.7681009769439697]], [[0.7132568359375]], [[0.7124687433242798]], [[0.6587449312210083]], [[0.7245516777038574]], [[0.8148982524871826]], [[0.6684139966964722]], [[0.6617448925971985]], [[0.5804407596588135]], [[0.7869804501533508]], [[0.7067265510559082]], [[0.6869311928749084]], [[0.7316954731941223]], [[0.780451774597168]], [[0.8454877138137817]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8cc1a66539dee9de64026d64a0395882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_be7e99b3892bfc2a0d3b54e6f165ec58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_25a2bbc70f5114d54c6eb4c3e40f952e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ab792128b6e61cb6b717bd2fa66c42fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0816437982f8bc924e2e41609e48669e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[-0.34831881523132324, -0.3267747759819031]], [[-0.4283560514450073, 0.12656813859939575]], [[0.40908509492874146, -0.2860028147697449]], [[0.0807693600654602, -0.30509230494499207]], [[-0.2860870063304901, 0.05865928530693054]], [[0.35354083776474, -0.2900979816913605]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_419345efde6fd279a358e98f49e39cab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[-0.034599900245666504, -0.1461290866136551]], [[-0.1180865466594696, 0.09195978194475174]], [[-0.041482940316200256, 0.022733867168426514]], [[0.07443660497665405, -0.2604130208492279]], [[-0.28827643394470215, -0.28109776973724365]], [[-0.09692291915416718, -0.1548452377319336]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ef02a1a97925635403fba8a445ebf9cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.34831881523132324, -0.3267747759819031]], [[-0.4283560514450073, 0.12656813859939575]], [[0.40908509492874146, -0.2860028147697449]], [[0.0807693600654602, -0.30509230494499207]], [[-0.2860870063304901, 0.05865928530693054]], [[0.35354083776474, -0.2900979816913605]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[-0.34831881523132324, -0.3267747759819031]], [[-0.4283560514450073, 0.12656813859939575]], [[0.40908509492874146, -0.2860028147697449]], [[0.0807693600654602, -0.30509230494499207]], [[-0.2860870063304901, 0.05865928530693054]], [[0.35354083776474, -0.2900979816913605]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c14de18fd199000d25ebc482fc962205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.034599900245666504, -0.1461290866136551]], [[-0.1180865466594696, 0.09195978194475174]], [[-0.041482940316200256, 0.022733867168426514]], [[0.07443660497665405, -0.2604130208492279]], [[-0.28827643394470215, -0.28109776973724365]], [[-0.09692291915416718, -0.1548452377319336]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[-0.034599900245666504, -0.1461290866136551]], [[-0.1180865466594696, 0.09195978194475174]], [[-0.041482940316200256, 0.022733867168426514]], [[0.07443660497665405, -0.2604130208492279]], [[-0.28827643394470215, -0.28109776973724365]], [[-0.09692291915416718, -0.1548452377319336]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_db1359dc964087c9136d3d870eb8b086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.10894569754600525], [0.08911313861608505], [0.12436171621084213], [0.03143560141324997], [0.02490701898932457], [0.09564900398254395]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[0.4026823937892914], [0.07231664657592773], [0.011129551567137241], [0.07582680135965347], [0.40325236320495605], [0.13427451252937317]]], dtype='float32').reshape([1, 6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9590d6f89c925af1031158bd58410732(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0033864504657685757], [0.003352757077664137], [0.00010585029667709023], [0.019867869094014168], [0.06527575105428696], [0.006096151191741228]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[0.4026823937892914], [0.07231664657592773], [0.011129551567137241], [0.07582680135965347], [0.40325236320495605], [0.13427451252937317]]], dtype='float32').reshape([1, 6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d1fd8c70bf440c6811fea268d43bd65d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8a3b13d9704ac86deb27bb32e0672ba0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 1, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c5f59339e0f6ede8ede7bedcf9dd6ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ad8d7d086ff864af5211aa6566dd9ddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a4cb4aec86efe73f5aa07a927a401877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_98a39c9389f30debe78de91d9f9ffcbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d87547ef0287c13296824099af3c041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29e04ac46758bf3ab7f17e1dbe67580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29e04ac46758bf3ab7f17e1dbe67580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29e04ac46758bf3ab7f17e1dbe67580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29e04ac46758bf3ab7f17e1dbe67580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d23f9141a4c3b28b79d1100a0976896(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e50832653eb5b4b72a91cc346ff92e0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1523a8a727bf3d522a0f9da6844e3700(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.34787654876709, 2.159102201461792, 1.9213602542877197, 2.2024269104003906, 2.1114273071289062, 2.1365914344787598, 1.959598422050476, 1.8337061405181885, 1.8550523519515991, 1.9595694541931152, 2.091510772705078, 2.2969653606414795, 1.895961880683899, 1.937199592590332, 2.1223039627075195, 2.2149817943573], dtype='float32').reshape([16]),
            paddle.to_tensor([0.6219276189804077, 0.8931844830513, 0.7872183322906494, 0.7686119079589844, 0.6709363460540771, 0.7716026306152344, 0.866905689239502, 0.9793623685836792, 0.9484323859214783, 0.5349599123001099, 0.8685113191604614, 0.550764799118042, 0.9109439849853516, 0.9845443964004517, 0.6136507987976074, 0.5252518057823181], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c67c60b88a4a77c53ede801e4df3d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.8714301586151123, 2.0343306064605713, 2.213071584701538, 2.2041175365448, 2.139944076538086, 2.1380159854888916, 1.9252699613571167, 2.2255287170410156, 1.8711376190185547, 2.096949577331543, 2.2395358085632324, 2.1725518703460693, 2.0819284915924072, 2.409768581390381, 2.231198310852051, 2.095301628112793], dtype='float32').reshape([16]),
            paddle.to_tensor([0.3780723512172699, 0.10681550204753876, 0.21278166770935059, 0.23138809204101562, 0.32906362414360046, 0.22839735448360443, 0.13309429585933685, 0.0206376351416111, 0.05156761035323143, 0.46504005789756775, 0.13148869574069977, 0.449235200881958, 0.08905600011348724, 0.015455592423677444, 0.3863491714000702, 0.4747481942176819], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68d9f7dfb97f6e7504f055d82d860fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5419362783432007, 0.5364436507225037, 0.49585777521133423, 0.5507045388221741, 0.5302027463912964, 0.5342292189598083, 0.4887573719024658, 0.46044811606407166, 0.46397048234939575, 0.505864143371582, 0.5277435779571533, 0.5602686405181885, 0.4781308174133301, 0.4861258566379547, 0.5410937666893005, 0.5395409464836121], dtype='float32').reshape([16]),
            paddle.to_tensor([0.368032842874527, 0.10307395458221436, 0.32535305619239807, 0.47889789938926697, 0.4783645570278168, 0.18116134405136108, 0.2584570348262787, 0.43919625878334045, 0.22095052897930145, 0.2511819303035736, 0.045693691819906235, 0.06585970520973206, 0.28214719891548157, 0.18147172033786774, 0.047013502568006516, 0.05115712434053421], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180e3ef2374da4d60fcff0448b4a7f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_af0c0243b4f98e8e3349b846611f0fa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.32124772667884827], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63258ce40165bddb5b802940feab2493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_982337694735f10f0b236a2b970ec976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_982337694735f10f0b236a2b970ec976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_982337694735f10f0b236a2b970ec976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d73c0b93fdc0adf0d84a9fc1783f580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2072802186012268], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5242ac030de9270829306e0119e38f42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.18305830657482147, 0.25759702920913696, 0.46923521161079407, 0.37422502040863037]]], dtype='float32').reshape([1, 1, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77d4f26afe1cb01c6183fafaaa57503f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_053c3e6058e11d1c222bfd71ef270604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.47799554467201233], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3af2468b581517491bc3faad9adc77a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.05867317318916321], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_879dbe7c74871c74ef5910984e697035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.029930606484413147], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cff5ed252f0a59b0c497fed595c414b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cff5ed252f0a59b0c497fed595c414b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cff5ed252f0a59b0c497fed595c414b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cff5ed252f0a59b0c497fed595c414b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31b48274dcd5e8f61211460521e2419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31b48274dcd5e8f61211460521e2419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31b48274dcd5e8f61211460521e2419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5a3eebd6da09ea72aeaae06144db2a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_409e7aa5d6426eb3f648feb2b6e74cb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.2647590637207031]], [[0.05532848462462425]], [[0.46784508228302]]], dtype='float32').reshape([3, 1, 1]),
        ]



class PrimitiveOp_f8c3d412899464bfbc225eb9c55d561d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1c12136e5b8d22b7be95df756f28130f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8c3d412899464bfbc225eb9c55d561d
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b2d3f5d0aac0982fb9a46ff33892f1a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa1356c9c9f299b29c2ce56f8e8fc32e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c3139c5b82e492bfbefd567659ac89f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3be4d1f2dd19f817a9d0f12ed990a613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3176971971988678], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3349b2d7e1f53a1f39f7736eada5d81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a34ac236da2cd7296464aa9c3d67736a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b527772f2389dcc2a7efb2e55c7c914e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_409850d31dc25d4be550d314c2cf873b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2829c59a382391a54571ed732ee6e05a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a728002b5ccb7a7206cf7ad3ec744ac9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e39a887b78cb2460a1376fc32b9060cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e39a887b78cb2460a1376fc32b9060cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e39a887b78cb2460a1376fc32b9060cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e39a887b78cb2460a1376fc32b9060cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b17714e530d4164ae2f7e17def424e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4734eea867e8a5788f63c1bcb4c35953(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3d78b9fdd9df8d56acc9aa2bded5d030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3d78b9fdd9df8d56acc9aa2bded5d030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3d78b9fdd9df8d56acc9aa2bded5d030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3d78b9fdd9df8d56acc9aa2bded5d030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3d78b9fdd9df8d56acc9aa2bded5d030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7e3ae703caf8acc5182266dab4f49bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7e3ae703caf8acc5182266dab4f49bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3d78b9fdd9df8d56acc9aa2bded5d030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7f1a1ad7772d4e7eadfe08b75f184117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80bfb6c9377203d662ab98f8fda86a96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dd4c93ef7b338536b22b511566a9bca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24196235835552216], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6c5ba64f4894d614d5c10bee70ab9160(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_db7776f434dda4c0cf85440e1e78929b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_be6beb54bdaadf7f3e41da74650657de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_75a6a2531c75ada5136fb76a009c360e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1614248901605606], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c4b85a2d5198046fbd85840785eab317(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 1, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13b1588f6e9741223c08664a57d8df13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13b1588f6e9741223c08664a57d8df13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13b1588f6e9741223c08664a57d8df13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7af2d4174f5334475826f5d31b62840d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7af2d4174f5334475826f5d31b62840d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7af2d4174f5334475826f5d31b62840d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7af2d4174f5334475826f5d31b62840d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e738b9b96f1a540c252fa38db5fd2986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_801a6a2bde13cdeceae96625867099c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_56ec365bf92c3b490693447565d29f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_696f63e4c62a7cd37ea73e8fc2fa4770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb8b6ec6950e4c4066e329aa43983c6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f0c65e7fe3d98905fc7769aaf13d930d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8798f97e43f0ae58953e2aeb44f9c860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8eab8735b4cdf331ba7f8761e9afd2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8eab8735b4cdf331ba7f8761e9afd2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8eab8735b4cdf331ba7f8761e9afd2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_631e38b7aa1d0fe84d89ec7a5c8aa93f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a5f61f4b7990bb04e177103c1ba277a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37f3594b4910713d020597ecc431b06c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ede582b6acf55247c0f8cdc42da718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ede582b6acf55247c0f8cdc42da718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ede582b6acf55247c0f8cdc42da718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ede582b6acf55247c0f8cdc42da718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0de9d5176a804fe69c7f211870754579(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7a91746fcd10805819c2f5a43435c1c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0721592903137207], [0.07056476175785065], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f1fc2f0ecaeefef8ce76955680f1d34d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.023881256580352783], [-0.17093104124069214], [-0.040314070880413055], [-0.23770397901535034], [0.19680941104888916], [-0.1749243587255478], [0.09875815361738205], [-0.23626181483268738], [-0.24265314638614655]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.10350233316421509], [-0.380374550819397], [-0.2866537570953369], [0.0721592903137207], [0.15432409942150116], [0.2023364007472992], [-0.22068245708942413], [-0.19273780286312103], [-0.3567226529121399]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f69f1f189af735ddb92a4d261136e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.06902246177196503], [-0.22078408300876617], [0.07766935229301453], [0.012395694851875305], [-0.15766564011573792], [-0.1496490091085434], [-0.18517188727855682], [-0.3871796131134033], [-0.1858700066804886]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.3100380301475525], [0.2617378234863281], [0.2190100997686386], [0.3838931620121002], [0.2870804965496063], [-0.20615726709365845], [0.08606493473052979], [-0.0019951313734054565], [-0.11980006098747253]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b27cb8116776b259e84a410d46512fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.21349862217903137], [-0.0710458755493164], [0.10224482417106628], [0.018052708357572556], [0.19680941104888916], [-0.07336302101612091], [0.09875815361738205], [-0.22516174614429474], [-0.1858700066804886]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.0028449296951293945], [0.2617378234863281], [0.2190100997686386], [0.3838931620121002], [0.37083983421325684], [0.2023364007472992], [0.14159324765205383], [-0.0019951313734054565], [-0.11980006098747253]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8549c7388c061c1897f4382fb48403b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_88cb30b2570b0f80d44010aad3ca79f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d7eb67e4bbcc0792c2cd5755376dae4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e70cbc65a0a34086ee724da5bd15697f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52898dc91fae291a3bfacf1bbe53e2e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_12b38ec5089cb522dd4c4e57c8eed0e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31835365295410156], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_12b38ec5089cb522dd4c4e57c8eed0e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31835365295410156], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_effdd7b6f64bd4a5d907c01d9cc48eb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_72dd0351faf391b37f8a2bfd1465103a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09322686493396759], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eeea747c86b046ce1a00368fde3bfe75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f78de7e79990e78d1b008f5da8bff78e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f91765b39bd5ed57d13f8b8309e3de37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_029b56c9593110dabb949a4eebf925c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_029b56c9593110dabb949a4eebf925c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_029b56c9593110dabb949a4eebf925c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_029b56c9593110dabb949a4eebf925c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_029b56c9593110dabb949a4eebf925c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d46c5814e02555a2379310afc7f63227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d46c5814e02555a2379310afc7f63227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_029b56c9593110dabb949a4eebf925c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b98e00d64b6523f8f2bb9a69470f516e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_92322378186ff8ff4b406a0aa3c774b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b59c496aaa08853ddb8fde71546ba2e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_702f930499f9d8b02b50b1db4c52b248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_702f930499f9d8b02b50b1db4c52b248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_702f930499f9d8b02b50b1db4c52b248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_096006a86e97a2713f11902f98f1e65e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2db90aad8765e96885340e0a33558153(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4930560290813446], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e284e09e0426db39970480aba2771403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67a4e49e7018d14b8b3163fe8c340d37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_288bce61607cd35ec11600e740d562c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_35205416b6efdbb103f130a1a015c5c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9a592c43058a6b29039a7b65d30b1dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8cc1a66539dee9de64026d64a0395882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2ac33d9fceb6459bb49cafe5783bae19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c143c2d36b50b9f882d3eacaa818c531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4368e4d4d196844a04fd2ee2859c3cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4306ba5ac915c281d66fba1a3fca09f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d55589cfb8ed3e32a4ed3d5b95e230b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_092ab289a333b4e31cbd2e65061fe339(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.462415486574173], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a92407a20ed8694427d831ac57143e55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_843be8890a0419f87a14ca723c84f9b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7343338131904602], dtype='float32').reshape([1]),
            paddle.to_tensor([0.27233320474624634], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_733fdd8f65104c8462e70c0e674b5706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7455248832702637], dtype='float32').reshape([1]),
            paddle.to_tensor([0.12875445187091827], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_996b03d4710c5afadef15444bec91531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4ec6addba0f0d1bd791f14ea68eecf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_674db8cde5e3d13a57323930ac2aec05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_17355833cecb8bd3c1b8f862e402dce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_22e7863ba23de80c4d218660f4264da4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 168, 168], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9c3612d2e33b59b33760b1a73713d5e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f6a08cb8f4a4f7f9bd5db0641ed9a8eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a7fb52331873956d59d21dc24cb6b2ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.01483928319066763], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_de6751153a82bd4679988db4fe2acb2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_30a2053f8f7ab44381e08e05ac233b62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3513548970222473], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b06b4a4cc76778e0f020aaa43817aff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0de9d5176a804fe69c7f211870754579(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffbd75d440faa196709ea2e71aca579a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2b601ccbb6d77d232d416d73e28aa8b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0ec7841af734e10357125abf937cfde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.019291505217552185, -0.27365997433662415, -0.3934049606323242, -0.16711992025375366, -0.018702328205108643, 0.26337409019470215], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0318598747253418, -0.007987573742866516, -0.21601513028144836, -0.025963157415390015, -0.06260842084884644, -0.07662948966026306], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7994c1a5a425fe9123e18f5d5c5db98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0006146249361336231, 0.002185879275202751, 0.08498142659664154, 0.004338960628956556, 0.0011709232348948717, -0.02018222212791443], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180fa63e8b15e1b08d462944a2b169f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0006146249361336231, 0.0, 0.0, 0.0, 0.0, -0.02018222212791443], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_438c0a62e26f71d46ffc604154ff2ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1580319106578827, 0.08440583944320679, 0.15580785274505615, 0.0, 0.0, 0.2714036703109741], dtype='float32').reshape([6]),
            paddle.to_tensor([0.24304825067520142, 0.0, 0.0, 0.16679298877716064, 0.09022623300552368, 0.0], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c96fedd38724aef6f91816dd93f2b2d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.019291505217552185, 0.04946009814739227, -0.3934049606323242, 0.016018692404031754, 0.011264503002166748, 0.3699343204498291], dtype='float32').reshape([6]),
            paddle.to_tensor([0.053903549909591675, 0.0039884597063064575, -0.061460524797439575, 0.1297207921743393, 0.06059029698371887, 0.05703727900981903], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_918326b4f86ee7089d4db17a32044c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.05832721292972565, 0.3405929505825043, -0.03126943111419678, 0.10864485800266266, -0.1369187980890274, -0.05729490518569946], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.05832721292972565, 0.3405929505825043, -0.03126943111419678, 0.10864485800266266, -0.1369187980890274, -0.05729490518569946], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f8b05c290b955360158be0a9e9370ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11661601066589355, -0.04659134894609451, 0.11697685718536377, 0.17422005534172058, 0.1380166858434677, -0.004758194088935852], dtype='float32').reshape([6]),
            paddle.to_tensor([0.11661601066589355, -0.04659134894609451, 0.11697685718536377, 0.17422005534172058, 0.1380166858434677, -0.004758194088935852], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f56624ab23fcf4eaabc6f362e7992b27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1580319106578827, 0.4075259268283844, 0.15580785274505615, 0.18313860893249512, 0.02996683120727539, 0.3779639005661011], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1580319106578827, 0.4075259268283844, 0.15580785274505615, 0.18313860893249512, 0.02996683120727539, 0.3779639005661011], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c9c80a8e9097db8b770b24399ed4763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2650919258594513, 0.011976033449172974, 0.1545546054840088, 0.32247692346572876, 0.213424950838089, 0.1336667686700821], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2650919258594513, 0.011976033449172974, 0.1545546054840088, 0.32247692346572876, 0.213424950838089, 0.1336667686700821], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_11ca1189d237ab37c2ecf4cb4d29728c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.09436476975679398, 0.951033890247345, 0.9186823964118958, 0.32256653904914856, 0.6005591154098511, 0.9353411793708801], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.232835590839386, 2.3465805053710938, 2.266756534576416, 0.7959005236625671, 1.4818192720413208, 2.3078603744506836], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_06f03019233eda98593c23a034fff7db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.02183729223906994, 0.6905632019042969, 0.6755805611610413, 0.2042846828699112, 0.47087711095809937, 0.683407723903656], dtype='float32').reshape([6]),
            paddle.to_tensor([0.021971477195620537, 2.231677532196045, 2.0824294090270996, 0.2567308843135834, 0.8899200558662415, 2.1586368083953857], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99b57530c5e9fc1e7438d632a4c47161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7587c4914963ef06fba98b39329b97c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_933657a556b874cc481a0954fbae7cc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5cdd0a5c170573c09cf2581b011a00dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5cdd0a5c170573c09cf2581b011a00dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5cdd0a5c170573c09cf2581b011a00dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5cdd0a5c170573c09cf2581b011a00dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5cdd0a5c170573c09cf2581b011a00dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_42529d90a9bc97431a7222dd002017b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_42529d90a9bc97431a7222dd002017b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5cdd0a5c170573c09cf2581b011a00dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f91765b39bd5ed57d13f8b8309e3de37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ced9c4ec59caa0ecbc1541cc6333a854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e6a87f9efd391ec359e8e657f995cffd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e45d478ebc09b0f5b899118271535c07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7f1a1ad7772d4e7eadfe08b75f184117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8989b5098a6e2417338f126c4c20d4a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_02e3b79c2ad678da7ef71f626e5a8fb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f6876c28ba3b8f3ee255555d924646f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.019055353477597237], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_082ea60ab4e6e66440bc5636c69d6dae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_862709e4458437e3e57e2ed5b0299ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b6a9a85758d388a814f05be1d72b9c46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77d4f26afe1cb01c6183fafaaa57503f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_113a9bc0918317f65bf6ce17e6e2c855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6a444c739c7fd4453b9c003d15d5fea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_23af230543be1fd6b73a8a3bae4a1c17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9ff821f9a0f1a0d267d848121bcc0920(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38422a7ea7c35fadf0d1d298b04ad0f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e59f7b9a91a0c7eaf8b4af5aee862ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2437657117843628], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c6df8900684c98e62955ce482bed7aed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_af0da6db58e08cbd0cbfd11c5b5bd61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1521999835968018, 2.1796727180480957, 2.1047537326812744, 2.159518241882324, 1.822599172592163, 1.9034342765808105, 2.1630988121032715, 1.9198029041290283, 2.1094632148742676, 2.0728089809417725, 1.9138891696929932, 1.9151036739349365, 1.9168546199798584, 1.9518407583236694, 1.93830144405365, 2.3321938514709473, 2.16713547706604, 2.033275842666626, 2.194241523742676, 2.034022092819214, 2.0413191318511963, 2.2838566303253174, 2.2300868034362793, 1.9731148481369019], dtype='float32').reshape([24]),
            paddle.to_tensor([0.8327269554138184, 0.7016278505325317, 0.8010571599006653, 0.5192842483520508, 0.6323118805885315, 0.5801719427108765, 0.8297624588012695, 0.833179235458374, 0.5878900289535522, 0.7204785346984863, 0.9652197360992432, 0.748534083366394, 0.6652483344078064, 0.9230014681816101, 0.8385258316993713, 0.8157482743263245, 0.8609439730644226, 0.8209357857704163, 0.8064541816711426, 0.7680544853210449, 0.8674399256706238, 0.9423626661300659, 0.8580211997032166, 0.7216925621032715], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4a174c2b5f098db626f3b749b09502b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.222297430038452, 2.2627525329589844, 1.849575400352478, 2.053041696548462, 2.192687511444092, 2.159512519836426, 2.076664686203003, 1.9242665767669678, 2.056380271911621, 2.302868604660034, 2.204963207244873, 2.3179593086242676, 2.261704444885254, 1.8650039434432983, 2.3297932147979736, 1.9024587869644165, 2.0811641216278076, 1.9169301986694336, 2.161933422088623, 2.0917773246765137, 2.026906728744507, 2.07029128074646, 2.2546253204345703, 2.260570526123047], dtype='float32').reshape([24]),
            paddle.to_tensor([0.16727307438850403, 0.29837217926979065, 0.19894284009933472, 0.48071572184562683, 0.3676881194114685, 0.4198280870914459, 0.17023754119873047, 0.16682079434394836, 0.41211000084877014, 0.27952146530151367, 0.03478027507662773, 0.25146594643592834, 0.3347516655921936, 0.07699854671955109, 0.16147416830062866, 0.18425174057483673, 0.1390560418367386, 0.17906422913074493, 0.1935458481311798, 0.2319454848766327, 0.13256007432937622, 0.057637352496385574, 0.14197881519794464, 0.27830740809440613], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67722abb5a8d04b57aca373bbcd1f557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5409813523292542, 0.5511153936386108, 0.5134969353675842, 0.5270832777023315, 0.4896690845489502, 0.5027357935905457, 0.5370961427688599, 0.4801368713378906, 0.5218967795372009, 0.5342788696289062, 0.4810032248497009, 0.5041020512580872, 0.5080734491348267, 0.4862886369228363, 0.5003793239593506, 0.56325364112854, 0.5387951731681824, 0.5031106472015381, 0.5469971299171448, 0.5118545293807983, 0.5098521709442139, 0.5678868293762207, 0.5583927035331726, 0.5132789611816406], dtype='float32').reshape([24]),
            paddle.to_tensor([0.09662725776433945, 0.2517531216144562, 0.3178901970386505, 0.2152634561061859, 0.0554405152797699, 0.45703259110450745, 0.03614778444170952, 0.01393285021185875, 0.002307600574567914, 0.07740283012390137, 0.11828718334436417, 0.4843491315841675, 0.24832983314990997, 0.2953934073448181, 0.3565485179424286, 0.32441121339797974, 0.2690918445587158, 0.3854125738143921, 0.3194684684276581, 0.002826559590175748, 0.13202808797359467, 0.41044580936431885, 0.3000727593898773, 0.32590940594673157], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60bd8a8178765b8fa68157b64b4dc374(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9687598f80eb1b3665c885e1eb1ff5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d61564a854d48b3c32d0bba1c1804d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4c8b19199ddf4f084ed43cb3e70f61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_674cf7ec3cc9630b256f7a9c2538487b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_674cf7ec3cc9630b256f7a9c2538487b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_674cf7ec3cc9630b256f7a9c2538487b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_674cf7ec3cc9630b256f7a9c2538487b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7af2d4174f5334475826f5d31b62840d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7af2d4174f5334475826f5d31b62840d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7af2d4174f5334475826f5d31b62840d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f67e1c043728b117321624a8ca995d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f67e1c043728b117321624a8ca995d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f67e1c043728b117321624a8ca995d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f67e1c043728b117321624a8ca995d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f67e1c043728b117321624a8ca995d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9482aeed256a689d566a3de56be055e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9482aeed256a689d566a3de56be055e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f67e1c043728b117321624a8ca995d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1f420eb2d4ecee2e859708f8d6f3cf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1f420eb2d4ecee2e859708f8d6f3cf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_445a022710021af2d5cb4d4ae2ccbf5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24376803636550903], [0.24617058038711548]]], dtype='float32').reshape([1, 2, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d04f25343dd602ccbbc19f7aac0fd124(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9b161e28f712eb1ea1773169963efbf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c5a0c2c829fc95e5bd13467cdc295253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_982337694735f10f0b236a2b970ec976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_982337694735f10f0b236a2b970ec976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_982337694735f10f0b236a2b970ec976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_982337694735f10f0b236a2b970ec976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_345faffd49c70dfdf97f948a23b15cf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 136, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d1fd8c70bf440c6811fea268d43bd65d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_572daef3ca46594dd20942f74f8f5656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_29987cec65b053d36063adfb5b84856f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07983528077602386], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c8c45872bf03349b471a3093bb75920e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89c36bdd37850b4971ee3f48170fc605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c7749614d1ea334f8039f540f2f1c957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1816418170928955, 1.8633242845535278, 2.1522037982940674, 2.2153337001800537], dtype='float32').reshape([4]),
            paddle.to_tensor([0.9042313694953918, 0.7810401320457458, 0.5276364088058472, 0.7198778390884399], dtype='float32').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60b60e4909c9546299d2e4d6519e919e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.207533359527588, 1.9045121669769287, 2.0253593921661377, 2.0253467559814453], dtype='float32').reshape([4]),
            paddle.to_tensor([0.09576865285634995, 0.21895988285541534, 0.47236356139183044, 0.28012216091156006], dtype='float32').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_51a1dd579e9fedb844732b9619c61478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5460303425788879, 0.46808570623397827, 0.5230717658996582, 0.5405285358428955], dtype='float32').reshape([4]),
            paddle.to_tensor([0.14503681659698486, 0.010800845921039581, 0.08474092930555344, 0.08312032371759415], dtype='float32').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe916c13b8d42e302b3d080a2cab9f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe916c13b8d42e302b3d080a2cab9f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe916c13b8d42e302b3d080a2cab9f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe916c13b8d42e302b3d080a2cab9f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05f85150ff366c1ba0cb3764c4734d9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a8ea5cb78cfa4778fddf3a439a5871a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4064a211017a5fcdf2cee542fa406560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d6e5fa54d18608cbb65a1dced9f71460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2b601ccbb6d77d232d416d73e28aa8b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8521392d3ca645f3346c6d4553b491ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d83ac101adc952374d22c20594177a3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3532690405845642], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1eeb9b6f0f20fabb3ca4bba981018876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0b15455d789d1cd911aaea6c15def69c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04022175073623657]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.32357287406921387]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc026026aaf0ae9b5c0e9b201d678261(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.23648566007614136]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.35454875230789185]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f8837bef47dd4801240fa286ad31b849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04022175073623657]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.30788999795913696]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9b161e28f712eb1ea1773169963efbf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a31f444d47524d01d518b2a8e6e936d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.02013292908668518], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.004770040512084961], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3829e1225eba948116986ba98bdc3316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.055064812302589417], [0.28067803382873535], [0.11762891709804535], [0.07574111223220825], [-0.1188349723815918], [-0.013302385807037354]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.07976526021957397], [-0.14685890078544617], [-0.15858013927936554], [-0.36050552129745483], [-0.013743840157985687], [-0.022245600819587708]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ca69ad638faf951042b3c664f830f38e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.16270190477371216], [-0.28220218420028687], [0.02013292908668518], [0.10196256637573242], [-0.014692038297653198], [0.24281780421733856]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3141138553619385], [0.06970024108886719], [-0.09054850041866302], [-0.2545846402645111], [0.10437272489070892], [-0.15421158075332642]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bdd5993025176145169ac6173c86af05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0719030350446701], [0.28067803382873535], [0.11762891709804535], [0.3329775035381317], [0.07311412692070007], [0.2764424681663513]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3891090750694275], [0.06970024108886719], [0.013089627027511597], [-0.23143820464611053], [0.1823386698961258], [0.07761061191558838]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4a5097f269f911eb5fd261b27c962751(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ca94b14a1814c04c691e7134626ab643(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8f11057eb0b002aaa9038b30565803b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e8d4c11fa2a9950864052ebb775d1f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ecf13d0c8af3654155035c5968579131(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5b79a0f5f555885ca005ea045cf407bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 184], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1fd6657e9887e60602d129fb4ecee2c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.6030163764953613]], [[0.7961981296539307]], [[0.6438075304031372]], [[0.7476810812950134]], [[0.8281542062759399]], [[0.754705548286438]], [[0.7412281036376953]], [[0.8092681169509888]], [[0.6864055395126343]], [[0.7921578288078308]], [[0.8171217441558838]], [[0.7205018997192383]], [[0.8045275807380676]], [[0.685869038105011]], [[0.7133346796035767]], [[0.8333665728569031]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_788c07cbada9cb79dc86ee45630506db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fa7228111aa9c21d7c92eb1d2b9bdb9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_207cb66e62286d278f579f3e034d24c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d4624be971395281f3348d7b5042079(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec888de54a45430ab57142c7afb0b1cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.29216843843460083], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cbce51e9a7ab3b2eb5220629ced53fff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b46e558de9f7da683afc2e1b16e8f927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e8434fa071132293306a8ce176445ab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e8434fa071132293306a8ce176445ab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8ccdc7477966dbf4cf7b60d2553764ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24834126234054565]]], dtype='float32').reshape([1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_39a5966664395c13d7b4a9c924cef8ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_810f2f96d2e08ba4c93d622d158634fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_436213cd8371db77e0dfc420c080c1b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b1c7bd3929d47edda21e8db7483b1250(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6e5973538dae092f95f6430541176fb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_90d1ed24e259969e4d3bac9d06d400c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13b1588f6e9741223c08664a57d8df13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13b1588f6e9741223c08664a57d8df13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13b1588f6e9741223c08664a57d8df13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13b1588f6e9741223c08664a57d8df13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e6055cd5a183c74c812596aae757825(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c710a3792e67cf709f6dd96710cb1802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e25f086ec1d7750982f77d0e06d44656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99ef619d75de616f4ebb7ea482c69fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d159dcdce92bef85a6371eb1f45ea98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66b3bc5612755e30bb3e34811f959395(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e34149a0fc917034c6b8e579dae1484(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec412de7df5f927099761cf43227c9b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bcd238191d1be9e42d5db88e153192cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b3bfeae364e8f08acb4ce36169ba537e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c6df8900684c98e62955ce482bed7aed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fbc50f0c14103f43a38bb2e73fc9fe24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dbcd85a28dd93c9407f4864f8d949ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f60c667b0241e05e213057e568f47a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f60c667b0241e05e213057e568f47a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f60c667b0241e05e213057e568f47a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_41186933064233e9b824ce56742f2103(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_886129294bcc26f060e00c256a939a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c8a8a67a979aeec8624a49d7b1a0aa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ad00744fbfd9323a67bf4fcb465f3471(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bbb054432bec96f1a2eee332e26c3b8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14183548092842102], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6a363a78addf4ea6b1503f8f84dc226a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dc0a8ea790de66b5cf902344912f3c47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8a6687314a1666704016ad75822d4436(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9d3a621ada0bfa8159e9095aa093995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dad3442e49d7d8b2cf4c445017c93b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dad3442e49d7d8b2cf4c445017c93b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dad3442e49d7d8b2cf4c445017c93b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dad3442e49d7d8b2cf4c445017c93b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dad3442e49d7d8b2cf4c445017c93b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61148415eca4b5d7411183e1e73fc9b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61148415eca4b5d7411183e1e73fc9b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dad3442e49d7d8b2cf4c445017c93b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8ca85a3143d85c33361ff1ac316a9bf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_958564af9ed246c4701356386b68370e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.9832037091255188], dtype='float32').reshape([1]),
            paddle.to_tensor([0.09283372014760971], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9fa2e0d07a2c51221fe404ebd33f8acd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8160135746002197], dtype='float32').reshape([1]),
            paddle.to_tensor([0.10435719788074493], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9060ab954a93c7505d9b1d055320de6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.854397177696228], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4671162962913513], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0457ee67122b2130cd91981323fda84e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8769496083259583], dtype='float32').reshape([1]),
            paddle.to_tensor([0.19629746675491333], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6e88bdcff1dea2d8d10685c3780fefed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8831506371498108], dtype='float32').reshape([1]),
            paddle.to_tensor([0.30381250381469727], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_33c9b0de2976e0cb0fe804ee5fdf2714(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8312440514564514], dtype='float32').reshape([1]),
            paddle.to_tensor([0.18074680864810944], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe98332a0102ab15f5015b54340b2699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7418454885482788], dtype='float32').reshape([1]),
            paddle.to_tensor([0.05117175728082657], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a229db73244ccfff899444aca713a105(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.9902271628379822], dtype='float32').reshape([1]),
            paddle.to_tensor([0.05169527232646942], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aea1f3f553c3535ff61693cd079f8003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8396027684211731], dtype='float32').reshape([1]),
            paddle.to_tensor([0.3299325704574585], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_81259b8361311032258f41dce78188df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68019eed6e7afdef9f3ca6eaa411efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68019eed6e7afdef9f3ca6eaa411efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68019eed6e7afdef9f3ca6eaa411efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68019eed6e7afdef9f3ca6eaa411efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4abd367406fc8384324b97f940d48fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aabef4c5efce04bb40f1ee4e80046941(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_56659ad2d8a15883d53abcb67234f557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_447aac738b0989c64fefd4301db92395(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.40025511384010315], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_447aac738b0989c64fefd4301db92395(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.40025511384010315], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b5fdccedee9857fec03c8806c3612ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1fe438861124f13c725ee8d8f95c06dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.32611703872680664], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9044d8d2dfe9a42e93a5586a621c5360(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df5961fb0c0f5b86c8586808e4275358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0ae67dfb4db16ad9e45502fbe13f8ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0ae67dfb4db16ad9e45502fbe13f8ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0ae67dfb4db16ad9e45502fbe13f8ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0ae67dfb4db16ad9e45502fbe13f8ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0ae67dfb4db16ad9e45502fbe13f8ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_034382baa3be9ef0da3b4eb3af57d79d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_034382baa3be9ef0da3b4eb3af57d79d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0ae67dfb4db16ad9e45502fbe13f8ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4ec6addba0f0d1bd791f14ea68eecf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_36ae65e88308366c195f46310d668d65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.21986964344978333]], [[0.2373102456331253]], [[0.24102632701396942]], [[0.2785630226135254]], [[0.47145286202430725]], [[0.2085183560848236]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c067a182db653cc7d98041e3ab7cea4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3440d6e7f195c00868928e578780c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3440d6e7f195c00868928e578780c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3440d6e7f195c00868928e578780c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3440d6e7f195c00868928e578780c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3440d6e7f195c00868928e578780c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df9e3c0e643e21e4910f290d6c486095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df9e3c0e643e21e4910f290d6c486095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3440d6e7f195c00868928e578780c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8b5d6331bba1306a28c890fd35f4cabf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a6e481c95a8c8f807ede787227efed7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5544dd10207a463a4c52e7ea9e763e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40c9d67b97400aeffbd71853214d578a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_75960fb36ae7b1af81dd1af1fc90b7a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1339c63c70e36c5ac6f77fc535c70a14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_153e45a8946ee5d702e6a26fd4874d06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c6797b5a05e82f6b056b9d71418c2db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5be1d7ce04a8445083abbb5e4591ec6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8989b5098a6e2417338f126c4c20d4a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3919d87949d4a9d7a9ebc0ae9fc21f58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8f11057eb0b002aaa9038b30565803b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_260948955ab43aa05075342c87f67b86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c280bed7344824b7762a9924fb7d91c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29e04ac46758bf3ab7f17e1dbe67580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29e04ac46758bf3ab7f17e1dbe67580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29e04ac46758bf3ab7f17e1dbe67580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_81a8c72261b95e0f62cd10522616fc6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fca340a03f0dc63414cdfb6c66e75b36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fca340a03f0dc63414cdfb6c66e75b36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2fc0a0094f515f1e8eb09e178dac456f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2016078680753708], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b2b8da6c5c38bd459725b795dd52191(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8eab8735b4cdf331ba7f8761e9afd2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8eab8735b4cdf331ba7f8761e9afd2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8eab8735b4cdf331ba7f8761e9afd2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8eab8735b4cdf331ba7f8761e9afd2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34ca3960bd4de3dd32152c21dfafc384(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_004ef6215b97d55fa392e242f5853d74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f76965b96a9a6070a3c10415410f256d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b98e00d64b6523f8f2bb9a69470f516e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b98e00d64b6523f8f2bb9a69470f516e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b98e00d64b6523f8f2bb9a69470f516e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2b8aa9e66a89b25c265e581c63352ee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ab8f70b2c050a7ee86e53fbb38be074f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7e6bda576d5ec3ddbd2e5318dcdb1b5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7e6bda576d5ec3ddbd2e5318dcdb1b5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3f0fb22b14a8db65dac163eb3afc4be2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.11621835827827454], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.22290584444999695], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_117cb4bf73a976943a54a54fbfe5624e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.06990808248519897], [0.19953130185604095], [0.2397157996892929], [-0.06576010584831238], [-0.17019523680210114]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.34975314140319824], [-0.14332401752471924], [0.01684647798538208], [0.24763968586921692], [-0.11009906232357025]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c2eafe978907301132493289e81bd962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.19966775178909302], [-0.07468777894973755], [0.11621835827827454], [0.10971325635910034], [0.14329633116722107]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2398657500743866], [0.2639036178588867], [-0.2593464255332947], [-0.402464359998703], [0.1149199903011322]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8380d0565bfb8cd30c72791d13c3bb15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.06990808248519897], [0.30127468705177307], [0.2397157996892929], [0.3520749509334564], [0.14329633116722107]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.3667130470275879], [0.2639036178588867], [0.01684647798538208], [0.24763968586921692], [0.209987074136734]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cbf4f01a2831428b37861cec8bce6814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c95e86d7edb30716407632b3fdff426b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2fda7caf84373f6c7e146acc96a8e9fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c18c8ffcd95c88918fad1a09ee2d7288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd1f95bb5e784c6d4308191662115631(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0ba7e8b04f21d5ca8d9473c991c176b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4c8b19199ddf4f084ed43cb3e70f61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d61564a854d48b3c32d0bba1c1804d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9687598f80eb1b3665c885e1eb1ff5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96b4d9789265aa21c2be90c5f2f9a9c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2afa241435484f1647a6e061f02e241f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int32').reshape([]),
            paddle.to_tensor(8, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bda743c7d3a31479c4a50d8d596e3c6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.011590427719056606], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6de4474a087fc47c7c493bc91339b736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dc0a8ea790de66b5cf902344912f3c47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d16b997ba6289f66ee0f7c994e24115c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37d0213c5905204291b6b30d7fa02aea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e08d73c7907c85a8682fb6b02ade2e00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ff75cf43355f38cb55b83d682f69204a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.27201327681541443], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7317c426c6efce2aec35bf06055acc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7317c426c6efce2aec35bf06055acc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7317c426c6efce2aec35bf06055acc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c830297c16690e3cc7eff08d73259294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4237370e4f4f6d607cf7f02c4008526d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cddceb97bbf73ab7fca4f6523b5cbd2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3beec96314edc0e4d69399871ff0954(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.05977138504385948], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3beec96314edc0e4d69399871ff0954(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.05977138504385948], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b08f91217d1b65d48918ced826c6f8e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_06605286f671d87362294774ac7cb7fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4519156515598297], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e39a887b78cb2460a1376fc32b9060cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e39a887b78cb2460a1376fc32b9060cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e39a887b78cb2460a1376fc32b9060cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_551a62b91296e5d15c53509f0ae3d8c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ede582b6acf55247c0f8cdc42da718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ede582b6acf55247c0f8cdc42da718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ede582b6acf55247c0f8cdc42da718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebe43fa5da4b58112ca3e2dfb4c3834d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebe43fa5da4b58112ca3e2dfb4c3834d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebe43fa5da4b58112ca3e2dfb4c3834d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e3c9c57bd77553b34959ff3087a81c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8cc1a66539dee9de64026d64a0395882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f9cb06afd1f944a7ce619912f66b262d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2758355438709259], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7004d0ffc159dd1f526eaec289535127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5afaf4001654d782ca4f86e3c30cb810(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_182b273622fab136ded7322940c440e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3c3486155d0f7e5cf4b076906cdfb2d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3c3486155d0f7e5cf4b076906cdfb2d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3c3486155d0f7e5cf4b076906cdfb2d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3c3486155d0f7e5cf4b076906cdfb2d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3c3486155d0f7e5cf4b076906cdfb2d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3913030d4134c98a301f004358b941f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3913030d4134c98a301f004358b941f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3c3486155d0f7e5cf4b076906cdfb2d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_750700295b5736232356262b5a4c716a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_984ffb78ba9376add8794763dab530d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a7109e5e1facdaf2d980b4fe4e7479d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a7109e5e1facdaf2d980b4fe4e7479d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a7109e5e1facdaf2d980b4fe4e7479d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a7109e5e1facdaf2d980b4fe4e7479d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a7109e5e1facdaf2d980b4fe4e7479d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b00d079dfb189f660244f6ddb9ff788(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b00d079dfb189f660244f6ddb9ff788(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a7109e5e1facdaf2d980b4fe4e7479d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0fad396d734d29d8522ff6ebf7f6d753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0fad396d734d29d8522ff6ebf7f6d753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0fad396d734d29d8522ff6ebf7f6d753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0fad396d734d29d8522ff6ebf7f6d753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0fad396d734d29d8522ff6ebf7f6d753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e7f1a2bc535011d635ce65608e335fb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e7f1a2bc535011d635ce65608e335fb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0fad396d734d29d8522ff6ebf7f6d753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1bb1ef52581963bb52c4f23198819df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2915319502353668], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1bb1ef52581963bb52c4f23198819df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2915319502353668], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a43acf25ac9a7c476ae22812d44a4350(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e6562846bcd7f88769bd4e717375c398(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.11829502135515213], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03daa1184e64444e1ace39f62994710c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5bc4de7890f5b619c295342df6ee99f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.7349135875701904]], [[0.9064015746116638]], [[0.8643157482147217]], [[0.8141332864761353]], [[0.9224852919578552]], [[0.891898512840271]], [[0.9145889282226562]], [[0.8929387927055359]], [[0.7257941365242004]], [[0.8682162165641785]], [[0.8355649709701538]], [[0.8789184093475342]], [[0.8609952926635742]], [[0.841895341873169]], [[0.8884779214859009]], [[0.8728126883506775]], [[0.8682401180267334]], [[0.8504614233970642]], [[0.866704523563385]], [[0.7024329900741577]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c8372f058ee4b3e91b95a44749b4d74d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_27e9129205a2c15dad3e936651f97f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9fad7faec7781072837f81ba9aed34ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ac47d30b53f100d981150a462dcb470c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_113835ffad837d2f5aa6fe921e172709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa47c5de95bb713e0372360f85cbe2e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3a87389f6acfb8105af3d10cd28a4a2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3464a2c3aa481a5de2665b0b3397255f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9585e9f19cd4817cf5f19dd8617e36be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea5efd702f4cedb946027c18181bbf29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68019eed6e7afdef9f3ca6eaa411efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68019eed6e7afdef9f3ca6eaa411efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68019eed6e7afdef9f3ca6eaa411efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e49db2506d2bb9fa03e6a6ddabb4f45d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20080240070819855], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2b7adb1d9c6933b77ca57ad501834b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e8c16f5f1042b4426bcaef43302da76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e24cf0a1582e185dd4e06117dd90ebcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc713adbaafc3a5a70963c7c66a60228(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8b5d6331bba1306a28c890fd35f4cabf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e8a4b349e7003096463036367f6abd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_de4deaea7dda2af08735523d167d8ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebb0afab2ec1b6dd001b186fe68e6e9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f432197224a4822094979d94a797986e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_646a8bd974ae0e68ae0241db200401f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c61901a8cb89254c1ec55101362492e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2192cc6acbf883c8b52946c269109578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ca3ac00bf090cdd799d7f85589e50637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3866167664527893], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c735bb73a880d0c17e47ce5fcc11e0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
    def get_inputs(self):
        return [
            paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.37492507696151733]], [[0.31706711649894714]]], dtype='float32').reshape([2, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_926a9dfbe7795add7d5fe35271ac06b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a8e6374bf3c50dcead728dd0fa121111(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_88407ccfcad06ccf3b5d3a641ec8eb9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b6140cb0e712efb327229bb7ebbedb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8dfa887fbf3c4f5891514b3fceed13b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5fc5679e63c82f46a5ed6d516ac9193(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_adea55ed341bebfd0a03eb7448f61eec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e9aeceb36396c0c7bb051ecd983cf96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_996b03d4710c5afadef15444bec91531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_842f192f9d67077955d60a77eab87ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5cae51af79abf1158f1d4a3a89d18f36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1815907955169678, 2.2769126892089844, 1.907932996749878, 2.0796091556549072, 2.110363006591797, 2.1431312561035156, 1.9416979551315308, 2.245042562484741, 2.209714651107788, 1.8555011749267578, 1.881988525390625, 2.0004849433898926, 2.0412535667419434, 1.860338807106018, 2.17122745513916, 1.93278968334198, 2.2222325801849365, 2.042781114578247, 2.1074957847595215, 2.1499881744384766], dtype='float32').reshape([20]),
            paddle.to_tensor([0.5526050925254822, 0.9161155819892883, 0.8250937461853027, 0.7532532811164856, 0.8525898456573486, 0.8517947196960449, 0.5593414306640625, 0.6254949569702148, 0.9594743251800537, 0.5187383890151978, 0.5623292922973633, 0.7030147314071655, 0.7970205545425415, 0.8406480550765991, 0.7951136827468872, 0.7482725381851196, 0.6160892248153687, 0.6457232236862183, 0.6575725078582764, 0.507962703704834], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_657f4d0630d559dfc733b48252df9139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.0350992679595947, 2.1021230220794678, 2.1693103313446045, 2.033270835876465, 1.9085214138031006, 1.9629747867584229, 2.2445266246795654, 1.951501488685608, 1.934584379196167, 2.248936414718628, 2.3342783451080322, 1.8598324060440063, 2.115058183670044, 2.0682835578918457, 1.9403040409088135, 2.1461122035980225, 2.099975109100342, 1.9092484712600708, 2.2656540870666504, 2.1662161350250244], dtype='float32').reshape([20]),
            paddle.to_tensor([0.4473949074745178, 0.08388441801071167, 0.17490623891353607, 0.2467467337846756, 0.14741015434265137, 0.14820529520511627, 0.4406585693359375, 0.37450501322746277, 0.04052567854523659, 0.48126161098480225, 0.4376707375049591, 0.2969852685928345, 0.2029794454574585, 0.15935193002223969, 0.2048863172531128, 0.25172749161720276, 0.38391077518463135, 0.35427677631378174, 0.34242749214172363, 0.49203726649284363], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62764f929d6bf0f1a0018d2e4d404a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5290127992630005, 0.5655626058578491, 0.48841238021850586, 0.5170438289642334, 0.5201523900032043, 0.5291078090667725, 0.5187854766845703, 0.5337774753570557, 0.5496411919593811, 0.511211633682251, 0.5199856758117676, 0.4896783232688904, 0.5140585899353027, 0.4733687937259674, 0.5309785604476929, 0.4966222047805786, 0.5438241362571716, 0.4988684058189392, 0.5404133796691895, 0.539493203163147], dtype='float32').reshape([20]),
            paddle.to_tensor([0.06382325291633606, 0.4575190544128418, 0.42838215827941895, 0.3507518470287323, 0.2551310062408447, 0.3879965841770172, 0.2360391467809677, 0.20912696421146393, 0.2905004322528839, 0.49082446098327637, 0.013807285577058792, 0.32814526557922363, 0.09752397239208221, 0.26693081855773926, 0.08003351092338562, 0.252758264541626, 0.3814542591571808, 0.08651839196681976, 0.21119371056556702, 0.016675109043717384], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_787e8e789f75c666222dbdfd16cb0aae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4306ba5ac915c281d66fba1a3fca09f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34bcfb0bab3dd0da4bd7b77b078f219d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6cae2f32295e183cb7b5b29e38e08554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_866a04eed06c600feff55d4d586540fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3d563f4e357bfe9c08713da4b4d31c7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c792bcaa81202288552f4f1418f98a7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e9fa8238e5208a03b5aaa23c1b046b07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.8911274075508118]], [[0.8892913460731506]], [[0.9355054497718811]], [[0.8367183208465576]], [[0.83566814661026]], [[0.8696799874305725]], [[0.8848949670791626]], [[0.8737568259239197]], [[0.9251277446746826]], [[0.8608295321464539]], [[0.8690536618232727]], [[0.8794706463813782]], [[0.8996880650520325]], [[0.8204582333564758]], [[0.8965553045272827]], [[0.7918619513511658]], [[0.8368157148361206]], [[0.9273790121078491]], [[0.8898094892501831]], [[0.9457544088363647]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c8372f058ee4b3e91b95a44749b4d74d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_27e9129205a2c15dad3e936651f97f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e8c16f5f1042b4426bcaef43302da76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9687598f80eb1b3665c885e1eb1ff5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d61564a854d48b3c32d0bba1c1804d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4c8b19199ddf4f084ed43cb3e70f61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_92b93f056018dfac7df6406a7e9d4169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_322987b5d210c3cea18b5a5987382c5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc713adbaafc3a5a70963c7c66a60228(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68df24610d8beb6cb58245609a429fad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fbc50f0c14103f43a38bb2e73fc9fe24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93d45dcae44bbead5224ffc8573cf1cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1be8b298b526e644fbd4d7587a3b813c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f432197224a4822094979d94a797986e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99d14f856526e690150ea007d0bb177b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d00cda4360dce0d641ccfe76ec3b041c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1eeb2615ab176a8a1a0a859273786b88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e8db5a91a546f06eaf6f9d2969b8668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7e5ec999a0953c9829213679f58a5944(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2e55e3546afcc3f397d2192ace91864a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cf43ceb41ed6b01aea1e7844c9fe2b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e8672b972d7ac7fcad31678ce6d1f2af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.024495869874954224]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ddb5b947c2fff43b17b363d7e0f8f858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.339305579662323], [-0.08985428512096405], [-0.1387719362974167], [0.19930914044380188]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.051651835441589355], [0.1649070680141449], [-0.02027750015258789], [0.042431771755218506]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa204298ad4541231a82cf99a429f2e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09980668127536774], [0.26724952459335327], [-0.19716008007526398], [0.13058194518089294]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.10653990507125854], [-0.19285885989665985], [-0.06860601902008057], [0.12275975942611694]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c6b24196eedb95c9d36bc3ac2509b68d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09980668127536774], [0.26724952459335327], [0.03494563698768616], [0.3053952157497406]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.22915056347846985], [0.1649070680141449], [0.0039511919021606445], [0.18009227514266968]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c5b546938e511c2ede3a3a1c3e5f9807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_81adde871c8e823266831f3035f4f803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_23af230543be1fd6b73a8a3bae4a1c17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_24fd49d5031f6696066dd6114d38c3d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cf43ceb41ed6b01aea1e7844c9fe2b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_21758aa2a0d039dcc0e880d9cf588f00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_14edacb60e712c06b62b6d521658ad16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52dd87e4dca3358f2ee52dfc4ef1238c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cd4a082c09ae58819b6ffab72c5d937f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1a7466534962b0a612c0d53f5dec9543(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b660637b02b5ab5407cc985ddb55d82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b660637b02b5ab5407cc985ddb55d82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b660637b02b5ab5407cc985ddb55d82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b660637b02b5ab5407cc985ddb55d82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b660637b02b5ab5407cc985ddb55d82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8296c6728420701b0a679b2ec2e93212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8296c6728420701b0a679b2ec2e93212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b660637b02b5ab5407cc985ddb55d82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b54599285bba30619d58cd55469f778(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_56e2fb5c9c5676d9514caf70b2794d8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0b3e5f67afc51b3cb235c3e417524d0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2bf88b2fcc25f2faec92bddb3576d0e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_87263c3b34529c51ca290d9c55fcde67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0a2c2ce88b473ba9ce3fce3f925f0a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b94ea20ce43805a3a55eca9b7764bfab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bfecdf9db17f3661a1a56b891220632d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.26912549138069153], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_21d0bb27ada4168d753e0254c4205882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c9ad89b6e8b736b8aae4892f647434f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 1, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_225f7face5f1f1e1a3a4b0c9a62c1920(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3996615409851074], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_225f7face5f1f1e1a3a4b0c9a62c1920(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3996615409851074], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1bb0f3f0c24529fe93e41ecb6538a487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_64ad0d1368b8c18144d38fbfe102b108(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.062410056591033936], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e444253e86fab847045a931de1b4b04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96717dea1e08ae16e4b310fffb7b49bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_048b4b85e2fa3b811895bfd279f7e580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4a822dcd1953dbb9535355c07633c1a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b3bfeae364e8f08acb4ce36169ba537e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8ae7a5c94d03b91cc763386ddd0bdc2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebe43fa5da4b58112ca3e2dfb4c3834d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebe43fa5da4b58112ca3e2dfb4c3834d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebe43fa5da4b58112ca3e2dfb4c3834d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebe43fa5da4b58112ca3e2dfb4c3834d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f76965b96a9a6070a3c10415410f256d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93151d990b832dc4bb03b8382433f80c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e70cbc65a0a34086ee724da5bd15697f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e36748a1422bc8c80033ecec490224f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e9aeceb36396c0c7bb051ecd983cf96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_af83c212de351e11b0d8b973ecaa6289(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_082ea60ab4e6e66440bc5636c69d6dae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a34ac236da2cd7296464aa9c3d67736a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5feca009dd83080f8fd3ee958f04aecc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e09791c1b9f89956ee193d39d2311393(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7317c426c6efce2aec35bf06055acc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d8c2e9323197aae553007cb789f72390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6a5ba4de4aa2523ea3f7b24f288af0df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.41445696353912354], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6e5973538dae092f95f6430541176fb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32e2e317924ab72d6f4f79c58e3948f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76127761220767ab72012abe4e5540b1
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_91f8e55d4be945712cbdef98d3e9db33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3598812520503998], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c5670afada48ca934ec9341e0417613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c66b5e808541b4865f6b44b595b1e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b6a9a85758d388a814f05be1d72b9c46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6de26e8b488402aea9e81e091a84753e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4050699770450592], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_984ffb78ba9376add8794763dab530d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_894df1063911ab67adb704d87e863d04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31b48274dcd5e8f61211460521e2419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31b48274dcd5e8f61211460521e2419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31b48274dcd5e8f61211460521e2419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31b48274dcd5e8f61211460521e2419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1eeb2615ab176a8a1a0a859273786b88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_801ad773442ada3e001b43035267b677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_387518b5eafb57a2476dd37d41810892(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_666bd2e85672a7a249356b1685a37415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf15f3e7c5e2b58d15d4869760940c0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed18cae95b5c13621bd0373be341adb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7e12b7ce88c6b2ebca79b925b0bc190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e195b068db5860ad94dd0ebb64eca70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_65d4dd1577dd8cf9fc980a51554d65e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2ba367eb94d659c6aabc923a36176967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3248227536678314], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6cf33cdab7e4ca073f98d4cdcc33d547(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_129198c9819ef85541b68601b6fd1b03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffbd75d440faa196709ea2e71aca579a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d2ac217f22f5d5534f633aa1f051e00e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.40715083479881287], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_382b659ffaa8149972086b9c4cdd1ba0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4555abc0611a87943e03789d78ac7d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4555abc0611a87943e03789d78ac7d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4555abc0611a87943e03789d78ac7d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bc52d9497816c303290b35858ce4b4d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bc52d9497816c303290b35858ce4b4d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bc52d9497816c303290b35858ce4b4d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bc52d9497816c303290b35858ce4b4d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bc52d9497816c303290b35858ce4b4d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3d4ad6b02d4771d27ecd7f40af041543(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3d4ad6b02d4771d27ecd7f40af041543(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bc52d9497816c303290b35858ce4b4d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea3afcd9b9d8424c5d883ec4a4c8d71
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9687598f80eb1b3665c885e1eb1ff5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d61564a854d48b3c32d0bba1c1804d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4c8b19199ddf4f084ed43cb3e70f61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396773a90d0ead0a0808c4fbbe81bf22
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ca08abb38b063805e5566035d747b4b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4200989902019501], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38ed482330abca07320923ac3b700ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b156014403960be5f44d36bb6e72cfa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4555abc0611a87943e03789d78ac7d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4555abc0611a87943e03789d78ac7d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4555abc0611a87943e03789d78ac7d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4555abc0611a87943e03789d78ac7d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e7dc14525835bfca2e3af432167a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2af88543875bfb8fb82453b6bd3c53de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_88407ccfcad06ccf3b5d3a641ec8eb9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8f4401fb326aad35f8984e7be007217a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_862dd1951ca207a3b6d6108a5a4bfc69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31590908765792847], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()