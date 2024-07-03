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
class PrimitiveOp_4ddcce8a180b8c33a60631653546bc50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3f444a16e823bb7d0f88307f2dc0a06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcce8a180b8c33a60631653546bc50
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4c693a4a2ff78966743a88daf201a080(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_31b28cd32470767ef967a29b208c7b9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c693a4a2ff78966743a88daf201a080
    def get_inputs(self):
        return [
            paddle.to_tensor([[4.7128376960754395, 4.874957084655762, 4.827646255493164, 4.325068473815918, 4.148613929748535, 4.263113975524902, 4.267441749572754, 4.787745475769043, 5.16745138168335, 4.418758392333984, 5.105741500854492, 5.473527431488037, 5.001097202301025, 4.5380330085754395, 5.232659339904785, 5.4267425537109375, 6.2028679847717285, 5.070359230041504]], dtype='float32').reshape([1, 18]),
        ]



class PrimitiveOp_c33ce085e6bf5d9ed0a32609edf6f9dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 23], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8aaba528d5d68a2ef1618634967accac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c33ce085e6bf5d9ed0a32609edf6f9dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[7.115499496459961, 6.420863628387451, 6.43386173248291, 6.495194911956787, 6.782318592071533, 6.629547595977783, 6.720054626464844, 6.494166851043701, 7.013171672821045, 6.801267147064209, 6.6948089599609375, 6.927262306213379, 7.087639808654785, 7.325156211853027, 5.409924030303955, 7.67278528213501, 7.237374782562256, 6.851604461669922, 6.978597640991211, 7.248482704162598, 7.14051628112793, 7.549689769744873, 6.527603626251221]], dtype='float32').reshape([1, 23]),
        ]



class PrimitiveOp_c5eef8ac6424fc66bad7653eaeda28b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61d8c5843ee71a4304fc62712513d2c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5eef8ac6424fc66bad7653eaeda28b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_d281a4a803bf8d92f7b832b52b52b462(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df92cfef751e75c1c138a97f2a850274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d281a4a803bf8d92f7b832b52b52b462
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4d787b651e173fdd72b62c54eb2369a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ce77de927db7aa386add706b4800d5a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d787b651e173fdd72b62c54eb2369a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_97a676267ac0217e9a726cdb750b02d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_abfa5feb67cf192fe83d2f6419a68987(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_a6e0ff18eb999e2b18c1238c9039b985(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_09cde8ab57639fa9db5b1f333ea0bf05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6e0ff18eb999e2b18c1238c9039b985
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_09cde8ab57639fa9db5b1f333ea0bf05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6e0ff18eb999e2b18c1238c9039b985
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_46c6c96cf66b0c7a3d16150e304f6ce0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 168, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_182d33db9ba31b1536ed2b4b293b153d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46c6c96cf66b0c7a3d16150e304f6ce0
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_54278e400b2be87ca05c602cb1ca8126(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 30, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8dbcf4f42ef894edd521a93fe743e5a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.932435989379883]], [[8.983295440673828]], [[7.800779342651367]], [[7.770794868469238]], [[7.657711982727051]], [[8.570398330688477]], [[7.985377788543701]], [[8.547874450683594]], [[6.691704750061035]], [[8.065608978271484]], [[8.797544479370117]], [[6.905008792877197]], [[7.936824798583984]], [[7.9261016845703125]], [[7.785417079925537]], [[7.818709850311279]], [[7.979026794433594]], [[8.503885269165039]], [[7.2908759117126465]], [[7.6919965744018555]], [[8.23678207397461]], [[8.094239234924316]], [[8.717683792114258]], [[7.794269561767578]], [[7.646255016326904]], [[8.356583595275879]], [[8.700400352478027]], [[7.808797359466553]], [[7.841263771057129]], [[8.376038551330566]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]



class PrimitiveOp_262ba2151df345eb3c5ae0bd348a8026(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eb534e011f7fd193685926449ddab4ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262ba2151df345eb3c5ae0bd348a8026
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_a324f85e300e87f041a35ba8fa34127e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66f128c8010ef1b87790aed3ae2520fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66f128c8010ef1b87790aed3ae2520fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66f128c8010ef1b87790aed3ae2520fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66f128c8010ef1b87790aed3ae2520fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66f128c8010ef1b87790aed3ae2520fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66f128c8010ef1b87790aed3ae2520fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66f128c8010ef1b87790aed3ae2520fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66f128c8010ef1b87790aed3ae2520fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd6df46a1cfba23274339f5962d9926a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd6df46a1cfba23274339f5962d9926a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd6df46a1cfba23274339f5962d9926a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd6df46a1cfba23274339f5962d9926a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd6df46a1cfba23274339f5962d9926a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd6df46a1cfba23274339f5962d9926a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd6df46a1cfba23274339f5962d9926a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd6df46a1cfba23274339f5962d9926a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8dbf55bf23098fcbb395faafe55f697c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8dbf55bf23098fcbb395faafe55f697c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8dbf55bf23098fcbb395faafe55f697c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8dbf55bf23098fcbb395faafe55f697c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8dbf55bf23098fcbb395faafe55f697c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8dbf55bf23098fcbb395faafe55f697c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8dbf55bf23098fcbb395faafe55f697c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8dbf55bf23098fcbb395faafe55f697c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a46b4f0d5d188955d59c82278df6b70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a46b4f0d5d188955d59c82278df6b70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a46b4f0d5d188955d59c82278df6b70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a46b4f0d5d188955d59c82278df6b70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a46b4f0d5d188955d59c82278df6b70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a46b4f0d5d188955d59c82278df6b70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a46b4f0d5d188955d59c82278df6b70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a46b4f0d5d188955d59c82278df6b70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_74d0080cabdd20368c3e36d951dcf3d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_74d0080cabdd20368c3e36d951dcf3d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_74d0080cabdd20368c3e36d951dcf3d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_74d0080cabdd20368c3e36d951dcf3d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_74d0080cabdd20368c3e36d951dcf3d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_74d0080cabdd20368c3e36d951dcf3d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_74d0080cabdd20368c3e36d951dcf3d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_74d0080cabdd20368c3e36d951dcf3d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f59d5bc23d10e2eef850c386338eb58f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.320115089416504]], [[7.914754390716553]], [[7.95361328125]], [[7.301033973693848]], [[7.661606311798096]], [[6.876945495605469]], [[7.888410568237305]], [[7.199883937835693]], [[7.128437042236328]], [[7.673472881317139]], [[6.833292007446289]], [[8.126873970031738]], [[7.337831974029541]], [[8.28213119506836]], [[7.123475551605225]], [[6.652795791625977]], [[7.81361722946167]], [[6.97962760925293]], [[6.899313449859619]], [[8.064953804016113]], [[7.767551898956299]], [[7.7723774909973145]], [[7.7174973487854]], [[7.75868034362793]], [[8.044079780578613]], [[8.732399940490723]], [[6.866262435913086]], [[7.858814716339111]], [[7.264653205871582]], [[8.07474136352539]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e10bfd0ec360e2d21613f0c79a8d420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ccabd4bdc8c15e1883d6b58f2ffcf4ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 5, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b34496f4e3905efb810f230deb72c2fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccabd4bdc8c15e1883d6b58f2ffcf4ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.8689806461334229]], [[1.5793933868408203]], [[1.1897251605987549]], [[1.9041125774383545]], [[1.7028452157974243]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]



class PrimitiveOp_db3a420cbd0dbb845b550b1a09e56109(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9f24be2c7574a602fd8075c37bd430e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db3a420cbd0dbb845b550b1a09e56109
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.5980634689331055]], [[2.7655820846557617]], [[2.7791013717651367]], [[3.011228561401367]], [[3.1274821758270264]], [[2.1656382083892822]], [[3.099547863006592]], [[2.400660753250122]], [[3.081286907196045]], [[2.728254556655884]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]



class PrimitiveOp_889d3db9c1718c4c87c2966a10bba689(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_724d888912d9f3f997ccf2be87c6634c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_889d3db9c1718c4c87c2966a10bba689
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_80ed7330d369c1f34d273814928826ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc1a7bb30e095b22f1014b4566f2a4ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.704773426055908]], [[5.791843891143799]], [[5.963242053985596]], [[5.590764999389648]], [[6.544337272644043]], [[5.998619556427002]], [[6.506511211395264]], [[5.367368221282959]], [[6.426480770111084]], [[6.108403205871582]], [[5.482844829559326]], [[5.846292972564697]], [[6.339420795440674]], [[6.519515037536621]], [[6.135477066040039]], [[5.573600769042969]], [[5.396278381347656]], [[5.879087448120117]], [[5.871366500854492]], [[6.803126335144043]], [[6.488256931304932]], [[5.959376811981201]], [[6.003581523895264]], [[6.510748863220215]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ac1d6bd59f07b7fb39f37ae57f4aadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_600a81d6fcfd3442889b3d9653a28b6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_bbde9be5291cab7a77e92af8048f17ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ad0afab41ade050dd03dc2e839091b94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbde9be5291cab7a77e92af8048f17ec
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e3ad8b760005d5e23980761deceac4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.729099273681641]], [[5.868844985961914]], [[5.271246433258057]], [[5.6339335441589355]], [[5.561883926391602]], [[6.020602226257324]], [[5.928430557250977]], [[5.490959644317627]], [[6.217313289642334]], [[5.175939083099365]], [[5.130229949951172]], [[5.410799980163574]], [[5.388720989227295]], [[5.670409202575684]], [[4.803288459777832]], [[5.071345329284668]], [[4.968140125274658]], [[4.7747883796691895]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_724d888912d9f3f997ccf2be87c6634c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_889d3db9c1718c4c87c2966a10bba689
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96c71381016af85563d23def58bd7742(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.410586357116699]], [[6.070959091186523]], [[6.3691086769104]], [[6.467810153961182]], [[5.7446370124816895]], [[6.573790550231934]], [[6.578603267669678]], [[6.422765254974365]], [[6.163666725158691]], [[6.709517955780029]], [[6.100464820861816]], [[6.599005222320557]], [[6.100033283233643]], [[6.477661609649658]], [[5.917914390563965]], [[6.493032932281494]], [[5.760428428649902]], [[6.31825590133667]], [[6.541898727416992]], [[5.9510064125061035]], [[6.197717189788818]], [[6.384490013122559]], [[6.556829929351807]], [[6.843836784362793]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0ae9794f8234e3548ee8dd15adbc120b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262ba2151df345eb3c5ae0bd348a8026
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_afb1a0e70110e434b78afd9f65a710fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_474a628119c59d06cc8fb2eb2a66ddb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a900fdf11eff94ab84a4d91ec15522ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_474a628119c59d06cc8fb2eb2a66ddb2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9010274410247803]], [[0.8791264295578003]], [[0.7772576808929443]], [[1.2044074535369873]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0ae9794f8234e3548ee8dd15adbc120b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262ba2151df345eb3c5ae0bd348a8026
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_d0a332f3f65a0baa0b24c0fb0a72a04c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 11, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_87ce8a48cd2f930aec0630ae8118607d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0a332f3f65a0baa0b24c0fb0a72a04c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.2572855949401855]], [[2.820734739303589]], [[2.724828004837036]], [[2.7143571376800537]], [[2.5331223011016846]], [[2.0727102756500244]], [[3.1677427291870117]], [[2.7983739376068115]], [[2.8266916275024414]], [[2.6178479194641113]], [[2.9327988624572754]]]], dtype='float32').reshape([1, 11, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61d8c5843ee71a4304fc62712513d2c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5eef8ac6424fc66bad7653eaeda28b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_724d888912d9f3f997ccf2be87c6634c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_889d3db9c1718c4c87c2966a10bba689
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_19359c54ba9fa814761aadfec38e9057(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e5829195b5969957bbc2b8cd268588e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19359c54ba9fa814761aadfec38e9057
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ca26c0dd770af20098437ad51fb530f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.317871570587158]], [[7.2048749923706055]], [[7.540952205657959]], [[7.160888195037842]], [[8.213842391967773]], [[7.61870002746582]], [[8.003351211547852]], [[7.220874786376953]], [[7.4271111488342285]], [[7.651928901672363]], [[7.9275431632995605]], [[8.04538345336914]], [[8.244603157043457]], [[7.6148905754089355]], [[7.728623867034912]], [[6.60276985168457]], [[8.527016639709473]], [[8.105579376220703]], [[7.484890460968018]], [[8.051274299621582]], [[7.621427536010742]], [[8.911153793334961]], [[7.644719123840332]], [[7.04080867767334]], [[7.401798248291016]], [[7.39521598815918]], [[7.999631881713867]], [[7.981103897094727]], [[7.7772536277771]], [[7.789159297943115]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_182d33db9ba31b1536ed2b4b293b153d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46c6c96cf66b0c7a3d16150e304f6ce0
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_015f60f0f7bb453319399c250296ff51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cbc5d5625170ea7c723848b0db9a62b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_015f60f0f7bb453319399c250296ff51
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_72466d6b0aac72c20d4bea6f95be44f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_0a30ddc94d09c5f2cfeb153b79834946(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_998f9463263c383f86f07efe873d3e65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a30ddc94d09c5f2cfeb153b79834946
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.879011154174805]], [[4.125312328338623]], [[4.7622294425964355]], [[3.8742642402648926]], [[3.6968116760253906]], [[4.492135047912598]], [[3.9997384548187256]], [[4.000436782836914]], [[4.1021928787231445]], [[4.1837239265441895]], [[4.110951900482178]], [[4.510783672332764]], [[4.736892223358154]], [[4.0170769691467285]], [[4.723196506500244]], [[4.257722854614258]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_21477458ce0127db95f5c638beed616f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_00c56c84749fe84060ac6ef515bb1b5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_24bd208a8b30167aa14a93a6b5110e16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0b3f6835f9fb139652b66a41978e8bb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_182d33db9ba31b1536ed2b4b293b153d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46c6c96cf66b0c7a3d16150e304f6ce0
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2869441ec7d1d254bf3b5a013fe440c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbde9be5291cab7a77e92af8048f17ec
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_c513ae7f47e626fe4801c31de815b10d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_558d75a863a543e1d813e0e31cebba1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c513ae7f47e626fe4801c31de815b10d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e5829195b5969957bbc2b8cd268588e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19359c54ba9fa814761aadfec38e9057
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fffc94f8ac1edaa7988798df83d31931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.063333511352539]], [[7.867304801940918]], [[8.279768943786621]], [[8.106486320495605]], [[7.967275142669678]], [[8.184263229370117]], [[8.04330825805664]], [[9.142849922180176]], [[8.232620239257812]], [[8.164482116699219]], [[8.263772010803223]], [[8.041988372802734]], [[9.274855613708496]], [[7.256785869598389]], [[8.023280143737793]], [[8.690044403076172]], [[8.282241821289062]], [[7.736745357513428]], [[8.28187084197998]], [[8.18561840057373]], [[7.383940696716309]], [[8.074625968933105]], [[8.256736755371094]], [[8.356426239013672]], [[8.775242805480957]], [[7.776888847351074]], [[8.092998504638672]], [[8.468311309814453]], [[8.004794120788574]], [[8.935215950012207]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]



class PrimitiveOp_7e04c7dfa61283a44e2d457562974ff9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2b777266b2716abf22b892a797401db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e04c7dfa61283a44e2d457562974ff9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2c026065d8c14e23899bdb4a05c680b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 218], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e79a8ac8e1db56f935e6f400640b05b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c026065d8c14e23899bdb4a05c680b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2ad8d2f145fdbb6389a6e940b3876077(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 25, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4328e3d3db30f32549a38ba240e8f8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ad8d2f145fdbb6389a6e940b3876077
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.039806365966797]], [[5.792285919189453]], [[5.926527500152588]], [[6.2489728927612305]], [[6.215531826019287]], [[6.358044624328613]], [[6.613975524902344]], [[6.986159801483154]], [[6.027462005615234]], [[5.791606426239014]], [[6.820392608642578]], [[6.442506313323975]], [[5.754520416259766]], [[6.488327980041504]], [[6.455815315246582]], [[6.31831169128418]], [[6.666366100311279]], [[6.836677551269531]], [[6.048222541809082]], [[6.6284098625183105]], [[6.555262088775635]], [[6.51648473739624]], [[5.54529333114624]], [[6.547698974609375]], [[7.419841289520264]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_724d888912d9f3f997ccf2be87c6634c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_889d3db9c1718c4c87c2966a10bba689
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4fdab81655a8e914d80e4f21b1d33cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_182d33db9ba31b1536ed2b4b293b153d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46c6c96cf66b0c7a3d16150e304f6ce0
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e35d8561d7b9961ef31d2921a8e43f41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c13c7c46462356ba91d80379403c783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e35d8561d7b9961ef31d2921a8e43f41
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ad7eda4a1aa1e7a6d8c599c1d597cc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6e0ff18eb999e2b18c1238c9039b985
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ad7eda4a1aa1e7a6d8c599c1d597cc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6e0ff18eb999e2b18c1238c9039b985
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fdcf158a51c1d35473da1e39b28f6445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262ba2151df345eb3c5ae0bd348a8026
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_9a52905ea4d1ceafc8927d22e04ee019(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38686e4211ddb2fa10fc19d11891df27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a52905ea4d1ceafc8927d22e04ee019
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_3c5047832242eaf2936e08bceb2f12c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_36ede82775bc928e42d342e72569cf8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5047832242eaf2936e08bceb2f12c5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.280507564544678]], [[5.606911659240723]], [[5.503065586090088]], [[5.575466156005859]], [[5.39021110534668]], [[6.354246616363525]], [[6.104427814483643]], [[6.293274879455566]], [[6.576566696166992]], [[5.8172149658203125]], [[5.8143815994262695]], [[6.955115795135498]], [[6.240105628967285]], [[5.636475086212158]], [[6.0233564376831055]], [[5.5612969398498535]], [[5.591333389282227]], [[5.680702209472656]], [[5.9278082847595215]], [[5.499972820281982]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_182d33db9ba31b1536ed2b4b293b153d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46c6c96cf66b0c7a3d16150e304f6ce0
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2b777266b2716abf22b892a797401db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e04c7dfa61283a44e2d457562974ff9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e5829195b5969957bbc2b8cd268588e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19359c54ba9fa814761aadfec38e9057
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_182d33db9ba31b1536ed2b4b293b153d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46c6c96cf66b0c7a3d16150e304f6ce0
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c4cf8440ae27300eb6e478673cb0a1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.024784088134766]], [[5.615845680236816]], [[4.925999641418457]], [[5.293482780456543]], [[5.037164211273193]], [[4.709914207458496]], [[5.823171138763428]], [[5.3064680099487305]], [[5.286867618560791]], [[5.390621662139893]], [[5.206210136413574]], [[5.673350811004639]], [[5.507486343383789]], [[5.8226518630981445]], [[5.315316200256348]], [[5.093769550323486]], [[4.747065544128418]], [[5.24083137512207]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_317ae5ce1a143b32e1ac921d709f50f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_015f60f0f7bb453319399c250296ff51
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3f444a16e823bb7d0f88307f2dc0a06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcce8a180b8c33a60631653546bc50
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ac87e20fd8cf8a217bb71b81ad412950(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_182d33db9ba31b1536ed2b4b293b153d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46c6c96cf66b0c7a3d16150e304f6ce0
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_11e50181806e25ed31483467fd1c390b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_558de5091dff68c1a5469940856324e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11e50181806e25ed31483467fd1c390b
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_c6faba843a864104047205ddfb8d0bbd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_49c8926fde6572ce7769aad0c8025a2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6faba843a864104047205ddfb8d0bbd
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_1f475e3f0b24fba0814821005955a9a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_65dbf3cb6e3fa2ee533ffbcdc17d4ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_65dbf3cb6e3fa2ee533ffbcdc17d4ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_49c8926fde6572ce7769aad0c8025a2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6faba843a864104047205ddfb8d0bbd
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_65dbf3cb6e3fa2ee533ffbcdc17d4ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_65dbf3cb6e3fa2ee533ffbcdc17d4ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_43ee4ee2844d9b98dabbbe24ba916497(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96ce5a2d51d5d84431c4ddbaff801067(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96ce5a2d51d5d84431c4ddbaff801067(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_649ba84948d95c0c868e3ac7f0f8af58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dad05cb0729707e54c6c4cfdc7d1b345(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dad05cb0729707e54c6c4cfdc7d1b345(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_5886281400e7eb7ed9c1f84512e93d36(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_557e0f5ada6937c12b9ec389828a8c7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5886281400e7eb7ed9c1f84512e93d36
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9913ffd08a0cf249a453be4cf1ce9058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9913ffd08a0cf249a453be4cf1ce9058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_557e0f5ada6937c12b9ec389828a8c7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5886281400e7eb7ed9c1f84512e93d36
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9913ffd08a0cf249a453be4cf1ce9058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9913ffd08a0cf249a453be4cf1ce9058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5f551b4276886ee6710beb28c466eb28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b15e945fae0158d3d8acd7ae79a32d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b15e945fae0158d3d8acd7ae79a32d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bcb14feaaa752cadb88f950caa1edadb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c09e2b0884e79c8cfbf17afb75ab342(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c09e2b0884e79c8cfbf17afb75ab342(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2d91316752f88d173c9cea18bb0ff959(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_506977c45023c397651b40b7751304bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d91316752f88d173c9cea18bb0ff959
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38686e4211ddb2fa10fc19d11891df27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a52905ea4d1ceafc8927d22e04ee019
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52b517cfa598952020ed1f5e2b7388ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.0720062255859375]], [[5.208503246307373]], [[5.06249475479126]], [[4.5323896408081055]], [[5.542492389678955]], [[5.169071197509766]], [[4.995449066162109]], [[4.422159671783447]], [[4.600342750549316]], [[4.8190999031066895]], [[4.795933246612549]], [[4.74271297454834]], [[4.916698932647705]], [[5.364006042480469]], [[5.094284534454346]], [[5.894742012023926]], [[4.947288513183594]], [[4.60883092880249]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eb534e011f7fd193685926449ddab4ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262ba2151df345eb3c5ae0bd348a8026
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63c9ad18d1b1359203384e3f15e9ca37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.289717674255371]], [[5.984675407409668]], [[6.481191635131836]], [[6.516063690185547]], [[6.890284061431885]], [[7.198267459869385]], [[5.973833084106445]], [[6.689905643463135]], [[6.024934768676758]], [[5.820860862731934]], [[6.1219377517700195]], [[6.105299472808838]], [[5.8296356201171875]], [[6.187517166137695]], [[6.363892555236816]], [[5.8717851638793945]], [[6.022918701171875]], [[6.813684463500977]], [[6.439299583435059]], [[6.676745891571045]], [[6.68774938583374]], [[6.478757858276367]], [[6.2004899978637695]], [[6.275086879730225]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f212d07453be99f73d9197a199be61a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1a7561583da1aed8036f2ea1d949323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.7462079524993896]], [[4.334737300872803]], [[4.708061218261719]], [[3.785792112350464]], [[4.199547290802002]], [[3.8263018131256104]], [[4.310879707336426]], [[3.833042860031128]], [[3.332792282104492]], [[4.134519100189209]], [[4.635512351989746]], [[4.0650634765625]], [[4.113148212432861]], [[4.272996425628662]], [[3.92016863822937]], [[4.341777801513672]], [[3.8617467880249023]], [[4.640167236328125]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]



class PrimitiveOp_d829e76cd8bc50ed20ee7bc38cf80572(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_369b609ebe2badf39f1378ff69a86e44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d829e76cd8bc50ed20ee7bc38cf80572
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f79012152da1f749f1dbf2666c81c671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d421f6da523c5a5cfc8f1a1c2fe264d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.40058708190918]], [[4.558745384216309]], [[5.096811294555664]], [[4.13717794418335]], [[4.5481414794921875]], [[4.9642510414123535]], [[4.2817182540893555]], [[4.441032886505127]], [[4.358351230621338]], [[3.9219157695770264]], [[4.180481433868408]], [[4.55796480178833]], [[4.259859085083008]], [[4.347965717315674]], [[4.469634532928467]], [[5.419482707977295]], [[4.937188148498535]], [[4.925285816192627]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_369b609ebe2badf39f1378ff69a86e44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d829e76cd8bc50ed20ee7bc38cf80572
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_238208547838856b8d51479a53e69c6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c9e3f9d6cdbf921b94ecfd7132d886f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_238208547838856b8d51479a53e69c6d
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f8ef500aa46c71018286eee3eafe15a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_41ba7185bd9fa5d8fa09975ae7515fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11e50181806e25ed31483467fd1c390b
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_058646190e08d9119017bb5f46afd5f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6faba843a864104047205ddfb8d0bbd
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93a38d392242259363cc626f53efd5d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93a38d392242259363cc626f53efd5d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_058646190e08d9119017bb5f46afd5f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6faba843a864104047205ddfb8d0bbd
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93a38d392242259363cc626f53efd5d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93a38d392242259363cc626f53efd5d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_98a674728c4a63a11269606898602811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c74843bdeeb299f31de6724b59c10094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c74843bdeeb299f31de6724b59c10094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a16f9b13b084ea76b2dd335a5a6b14f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2acbe6d70819f6e710bb62ef1458bfb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2acbe6d70819f6e710bb62ef1458bfb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37f91790a08b7d4f6407bdce67c66612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5886281400e7eb7ed9c1f84512e93d36
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ca739a7ad1a3a34eb1e356b173e5ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ca739a7ad1a3a34eb1e356b173e5ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37f91790a08b7d4f6407bdce67c66612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5886281400e7eb7ed9c1f84512e93d36
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ca739a7ad1a3a34eb1e356b173e5ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ca739a7ad1a3a34eb1e356b173e5ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_198d4c48309b2e59d3f0e5fa579db7e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_698f5aa4638e1a6d7458ef5ad6677dad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_698f5aa4638e1a6d7458ef5ad6677dad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a2ad5a59b52e98d9ec65e3c0d5e8ce29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_db7aa3318d96bfafb12628739f41435b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_db7aa3318d96bfafb12628739f41435b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eec751309aaaf735c626809184d69df7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d91316752f88d173c9cea18bb0ff959
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9793a83985a5545bcc7f86f6ff54b144(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d787b651e173fdd72b62c54eb2369a9
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_558d75a863a543e1d813e0e31cebba1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c513ae7f47e626fe4801c31de815b10d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b75d034f5b719d4e1e0d7f77bdcda92c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262ba2151df345eb3c5ae0bd348a8026
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a19371a019153eeebfcc1c2002fdb451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93beaed7fdc06041821f9a131a281c1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_015f60f0f7bb453319399c250296ff51
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fdcf158a51c1d35473da1e39b28f6445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262ba2151df345eb3c5ae0bd348a8026
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9bb2764386a75c36ad12f7d8c3290f16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9bb2764386a75c36ad12f7d8c3290f16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f81e8256de2c28efc0e7fcb893a4059d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f81e8256de2c28efc0e7fcb893a4059d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e1590cae79824c040f2f01766bb8510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e1590cae79824c040f2f01766bb8510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e1590cae79824c040f2f01766bb8510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_b40cf8fc93480e1e685d24ce493ca7b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9d0583346cab4964d511f7bea2c765c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40cf8fc93480e1e685d24ce493ca7b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9d0583346cab4964d511f7bea2c765c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40cf8fc93480e1e685d24ce493ca7b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9d0583346cab4964d511f7bea2c765c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40cf8fc93480e1e685d24ce493ca7b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_497c5e27c889d11ca71ed736c31a7184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40cf8fc93480e1e685d24ce493ca7b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_497c5e27c889d11ca71ed736c31a7184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40cf8fc93480e1e685d24ce493ca7b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_497c5e27c889d11ca71ed736c31a7184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40cf8fc93480e1e685d24ce493ca7b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_24a5e5ec0efd252971fc110b91e7dd9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32fd52216a73956c5b63e51534975c4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24a5e5ec0efd252971fc110b91e7dd9d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32fd52216a73956c5b63e51534975c4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24a5e5ec0efd252971fc110b91e7dd9d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_487403fd912625ce73453cdfb450204f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_45425b1b0a86fbca4b3e7ccd6185163d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40cf8fc93480e1e685d24ce493ca7b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c182c57e5715cc6886c3c6d56c1697ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b92c609e107c5ccffc32e9669f73b874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ff20e0fce4938185b1779050269a39de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0702b4c9774d20736f7f9f999bad9044(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c73a7b1021ac3b285b10600f9e8b0664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c4b1be2fd0af04f55cd1b59df80c92d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_182d33db9ba31b1536ed2b4b293b153d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46c6c96cf66b0c7a3d16150e304f6ce0
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e11e636c0b45a46f33f921557cc3481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fb2be2969a9b9760ba8e8e05172f0e24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbde9be5291cab7a77e92af8048f17ec
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9cc0a6ff1a3694162374c0eff0b20b79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_015f60f0f7bb453319399c250296ff51
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_442b684365dc084debd9943b580299ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4a10be309cfa8ce66e445a94f5fa9b9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.744770050048828]], [[6.469620227813721]], [[4.977606296539307]], [[5.104647636413574]], [[5.340109825134277]], [[4.746143817901611]], [[5.46645975112915]], [[4.978821277618408]], [[4.684392929077148]], [[5.801762580871582]], [[5.27894926071167]], [[4.450350284576416]], [[5.514953136444092]], [[4.709138870239258]], [[4.926785469055176]], [[5.040286540985107]], [[5.034231662750244]], [[4.999880313873291]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38686e4211ddb2fa10fc19d11891df27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a52905ea4d1ceafc8927d22e04ee019
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e4ade6b2ee6ba32658f59d6f8e8743cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a74c6d57d87f24f14bca612b6b95d7c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4ade6b2ee6ba32658f59d6f8e8743cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61d8c5843ee71a4304fc62712513d2c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5eef8ac6424fc66bad7653eaeda28b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_632f63acbb546180c3fa6ed431c2d012(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15f69394867058012629da7cc26e80be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a30ddc94d09c5f2cfeb153b79834946
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.15363883972168]], [[4.262726783752441]], [[4.043941974639893]], [[3.477867603302002]], [[3.7144417762756348]], [[3.9724888801574707]], [[4.637231826782227]], [[3.9909329414367676]], [[3.735192060470581]], [[3.4399991035461426]], [[4.4346394538879395]], [[4.366098880767822]], [[3.759702205657959]], [[4.257101535797119]], [[4.576810836791992]], [[3.867220401763916]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a71067182d1fb86582318724cb2b9c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_238208547838856b8d51479a53e69c6d
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e5829195b5969957bbc2b8cd268588e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19359c54ba9fa814761aadfec38e9057
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_deafe51a4b36bad2478ae3fa8d367afd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.576985836029053]], [[4.620308876037598]], [[4.691135406494141]], [[4.501779556274414]], [[4.308021068572998]], [[4.544992446899414]], [[4.8089189529418945]], [[5.429499626159668]], [[4.519501686096191]], [[5.360790252685547]], [[4.243905067443848]], [[4.462162971496582]], [[4.71541166305542]], [[4.232010841369629]], [[4.425318717956543]], [[5.0960869789123535]], [[4.9816975593566895]], [[4.895888805389404]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68ec87d59ec42152d258bf13c18996c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_474a628119c59d06cc8fb2eb2a66ddb2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0201356410980225]], [[1.2915623188018799]], [[0.6410092115402222]], [[1.0970127582550049]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c09885001a3898b40360d1e3d17ca71b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11e50181806e25ed31483467fd1c390b
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1629b82f9b9d264df59e57e0310df544(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6faba843a864104047205ddfb8d0bbd
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_28845c868806081964769ba364d6a258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_28845c868806081964769ba364d6a258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1629b82f9b9d264df59e57e0310df544(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6faba843a864104047205ddfb8d0bbd
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_28845c868806081964769ba364d6a258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_28845c868806081964769ba364d6a258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_51f6d1f6a75dfc26bd6e000f0483347a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93bab211153fc9a3f5aa72a859b07fca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93bab211153fc9a3f5aa72a859b07fca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5aa80fe33952923fed049f8e7b4e0674(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493220f4d963512b494ce8e943c2975b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493220f4d963512b494ce8e943c2975b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d183eb4a30dad0de704fac334d8b635e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5886281400e7eb7ed9c1f84512e93d36
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e9c86937873075c216681670b3e723ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e9c86937873075c216681670b3e723ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d183eb4a30dad0de704fac334d8b635e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5886281400e7eb7ed9c1f84512e93d36
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e9c86937873075c216681670b3e723ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e9c86937873075c216681670b3e723ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e788072bbb64fe4659b4c757729a032d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8944ed069b6fa8021c8b68a6ce5c421d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8944ed069b6fa8021c8b68a6ce5c421d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a35fa6e16806273f49b7b008d6c10065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3927e24ee32d3a21801ee19b7c9e808f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3927e24ee32d3a21801ee19b7c9e808f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_998531eb6224066907af72dcfe410c09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d91316752f88d173c9cea18bb0ff959
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38686e4211ddb2fa10fc19d11891df27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a52905ea4d1ceafc8927d22e04ee019
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c13c7c46462356ba91d80379403c783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e35d8561d7b9961ef31d2921a8e43f41
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3ec8b75aa7b4bbc942d38f923bc742d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbde9be5291cab7a77e92af8048f17ec
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_78c97351a28248394bc67eb13bf250f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 168], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_456e5d883195310ae232b35385bd8fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c97351a28248394bc67eb13bf250f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_72fe2a48ee5468f7c5c2d180a930b506(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93e2d925a5b151abf25605c695763c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72fe2a48ee5468f7c5c2d180a930b506
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b75d034f5b719d4e1e0d7f77bdcda92c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262ba2151df345eb3c5ae0bd348a8026
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38686e4211ddb2fa10fc19d11891df27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a52905ea4d1ceafc8927d22e04ee019
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e5829195b5969957bbc2b8cd268588e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19359c54ba9fa814761aadfec38e9057
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_369b609ebe2badf39f1378ff69a86e44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d829e76cd8bc50ed20ee7bc38cf80572
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_182d33db9ba31b1536ed2b4b293b153d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46c6c96cf66b0c7a3d16150e304f6ce0
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5f9f0fecde1a59970b2847dea955bb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5047832242eaf2936e08bceb2f12c5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.691684722900391]], [[5.826727390289307]], [[5.569563865661621]], [[4.746211528778076]], [[5.540277004241943]], [[5.252917766571045]], [[4.920151233673096]], [[4.923593044281006]], [[4.93858528137207]], [[5.285863399505615]], [[5.395374298095703]], [[5.400188446044922]], [[5.340855121612549]], [[5.122163772583008]], [[5.982247352600098]], [[6.027462005615234]], [[5.638346195220947]], [[5.619996070861816]], [[5.0743632316589355]], [[5.145237445831299]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]



class PrimitiveOp_f7c9e485d398baed71cdc1ede6777d74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f901dd80ed806474c7409d5ce2a5618d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7c9e485d398baed71cdc1ede6777d74
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_0aaf3f9c9f43c1b89281b8ed63f65d42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_004d362657039df37e7d3bed7cd928f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aaf3f9c9f43c1b89281b8ed63f65d42
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.3242204189300537]], [[2.5504298210144043]], [[3.132251262664795]], [[2.8135931491851807]], [[3.2420735359191895]], [[2.9025022983551025]], [[3.072528839111328]], [[2.5479371547698975]], [[2.4486899375915527]], [[3.069793939590454]], [[3.234339475631714]], [[3.066904067993164]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_001d1ace28c8005c430bf4ef1ae84dee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5047832242eaf2936e08bceb2f12c5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.2184977531433105]], [[5.2667107582092285]], [[5.434055328369141]], [[5.131888389587402]], [[4.453257083892822]], [[6.212052345275879]], [[5.52595853805542]], [[4.89768123626709]], [[4.918511867523193]], [[5.475644588470459]], [[5.82695198059082]], [[5.78203010559082]], [[5.754968643188477]], [[5.206923484802246]], [[5.637141704559326]], [[5.170568943023682]], [[5.467113494873047]], [[4.992288589477539]], [[4.318904876708984]], [[5.6925435066223145]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d0905b3de43ee3ff7677f9a233e2d7c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0a332f3f65a0baa0b24c0fb0a72a04c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.4854507446289062]], [[2.4385039806365967]], [[2.5216965675354004]], [[2.1648199558258057]], [[2.632795810699463]], [[2.3447458744049072]], [[3.06015944480896]], [[2.8932671546936035]], [[2.719749927520752]], [[2.519822835922241]], [[2.353712797164917]]]], dtype='float32').reshape([1, 11, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e5829195b5969957bbc2b8cd268588e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19359c54ba9fa814761aadfec38e9057
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93e2d925a5b151abf25605c695763c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72fe2a48ee5468f7c5c2d180a930b506
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efb35a7c332a288b46efb0bbf27af76b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_20328261b99abcac38b3c71db66df47f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 14, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1519de39d358827219dda85b4ea4ca12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20328261b99abcac38b3c71db66df47f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.3554539680480957]], [[3.217783212661743]], [[2.8873744010925293]], [[3.1918511390686035]], [[3.068450450897217]], [[2.8076624870300293]], [[3.5078229904174805]], [[3.385564088821411]], [[3.763841390609741]], [[3.3057873249053955]], [[3.5406668186187744]], [[3.7813518047332764]], [[3.6586873531341553]], [[3.3288607597351074]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]



class PrimitiveOp_85ed9acc28e00af511e7803f56e3294e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4bbff5adc6304be723ac83be663b66d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85ed9acc28e00af511e7803f56e3294e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_600a81d6fcfd3442889b3d9653a28b6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61d8c5843ee71a4304fc62712513d2c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5eef8ac6424fc66bad7653eaeda28b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ccf338577ed43b8c544054147a42fcca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5047832242eaf2936e08bceb2f12c5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.097700119018555]], [[5.456888675689697]], [[5.51584529876709]], [[5.612609386444092]], [[5.830142974853516]], [[5.613444805145264]], [[5.420537948608398]], [[6.005110263824463]], [[5.142417907714844]], [[5.411586284637451]], [[5.896563529968262]], [[4.884195327758789]], [[6.2467546463012695]], [[5.234903812408447]], [[5.787017822265625]], [[5.4528889656066895]], [[5.498476028442383]], [[5.138840675354004]], [[5.521658897399902]], [[5.879971981048584]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04acc47440cb564ebe3112e01d7cb447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04acc47440cb564ebe3112e01d7cb447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04acc47440cb564ebe3112e01d7cb447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04acc47440cb564ebe3112e01d7cb447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_8276864881ce911fc9b794b28454fd15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b0e239e87243a4b1901e460d1d037e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8276864881ce911fc9b794b28454fd15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[31571.841796875]], [[35344.24609375]], [[36648.8671875]], [[35764.453125]], [[35387.73046875]], [[32963.13671875]]], [[[31131.6484375]], [[34854.49609375]], [[36155.8203125]], [[35275.0234375]], [[34907.05078125]], [[32515.74609375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df35c3c570e83308da19fd98c6d716d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8276864881ce911fc9b794b28454fd15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[36937.890625]], [[44298.43359375]], [[38864.83984375]], [[40726.38671875]], [[46830.0546875]], [[40810.7578125]]], [[[34079.921875]], [[40872.62890625]], [[35855.78515625]], [[37574.31640625]], [[43202.19140625]], [[37654.83984375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0afa2aeedc41ccde49b7ef7b4204465d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8276864881ce911fc9b794b28454fd15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[41162.078125]], [[43691.71875]], [[48400.57421875]], [[33811.140625]], [[47499.08984375]], [[44477.12890625]]], [[[37595.48828125]], [[39901.8125]], [[44203.8828125]], [[30882.15234375]], [[43378.43359375]], [[40617.95703125]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f7cca776cda56dfbe21c12b661b28172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8276864881ce911fc9b794b28454fd15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43057.671875]], [[50181.33203125]], [[45159.3359375]], [[42627.359375]], [[43513.625]], [[43262.671875]]], [[[39031.8515625]], [[45485.43359375]], [[40927.40625]], [[38637.046875]], [[39445.37109375]], [[39215.1484375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_625e39569a696b73a64825395d4207a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_625e39569a696b73a64825395d4207a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_625e39569a696b73a64825395d4207a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_625e39569a696b73a64825395d4207a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_625e39569a696b73a64825395d4207a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_625e39569a696b73a64825395d4207a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_625e39569a696b73a64825395d4207a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_625e39569a696b73a64825395d4207a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_744a4862f6ed7685aa950d1b2266fc39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_744a4862f6ed7685aa950d1b2266fc39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_744a4862f6ed7685aa950d1b2266fc39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_744a4862f6ed7685aa950d1b2266fc39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_744a4862f6ed7685aa950d1b2266fc39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_744a4862f6ed7685aa950d1b2266fc39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_744a4862f6ed7685aa950d1b2266fc39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_744a4862f6ed7685aa950d1b2266fc39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4586088d9a8789706d22c80ac8e24e05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4586088d9a8789706d22c80ac8e24e05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4586088d9a8789706d22c80ac8e24e05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4586088d9a8789706d22c80ac8e24e05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4586088d9a8789706d22c80ac8e24e05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4586088d9a8789706d22c80ac8e24e05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4586088d9a8789706d22c80ac8e24e05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4586088d9a8789706d22c80ac8e24e05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e5829195b5969957bbc2b8cd268588e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19359c54ba9fa814761aadfec38e9057
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_369b609ebe2badf39f1378ff69a86e44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d829e76cd8bc50ed20ee7bc38cf80572
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_64c8d908fbcb1ed03ae2cd82db062bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.533145904541016]], [[7.354318141937256]], [[7.447439670562744]], [[7.0797119140625]], [[7.624172687530518]], [[7.093131065368652]], [[6.802231311798096]], [[6.923765182495117]], [[7.4682817459106445]], [[7.509212970733643]], [[6.72806453704834]], [[7.412635326385498]], [[7.149740219116211]], [[7.231971740722656]], [[6.481606483459473]], [[8.51262378692627]], [[7.945773601531982]], [[7.963558673858643]], [[7.250692367553711]], [[6.296770095825195]], [[7.527583599090576]], [[7.976482391357422]], [[7.857499599456787]], [[7.265157699584961]], [[7.281493186950684]], [[6.802944660186768]], [[8.28808879852295]], [[6.954543113708496]], [[7.341640472412109]], [[7.064842224121094]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d23190778c44c0aa122cf7da238b163d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.124375343322754]], [[7.7985639572143555]], [[8.486802101135254]], [[8.122688293457031]], [[8.025605201721191]], [[7.954373359680176]], [[8.63216781616211]], [[8.88422679901123]], [[7.932723522186279]], [[9.44963264465332]], [[9.011527061462402]], [[8.29687213897705]], [[8.6183443069458]], [[8.34325885772705]], [[8.56541633605957]], [[7.821725845336914]], [[7.581669807434082]], [[8.553250312805176]], [[7.7252655029296875]], [[7.795560836791992]], [[8.779960632324219]], [[8.769819259643555]], [[8.215607643127441]], [[7.944150924682617]], [[7.799108028411865]], [[8.4075345993042]], [[8.656457901000977]], [[8.188000679016113]], [[8.324912071228027]], [[8.560117721557617]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2fc0dd9dbac1e8cc69b8501f2201fa4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_87338ca571b16a6b2f1855498bd40a1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.451113224029541]], [[7.2594804763793945]], [[7.049568176269531]], [[7.04967737197876]], [[7.738953590393066]], [[6.293715476989746]], [[7.056695461273193]], [[6.882185459136963]], [[7.1396164894104]], [[6.536636829376221]], [[7.101452350616455]], [[6.9430952072143555]], [[6.751807689666748]], [[6.944225311279297]], [[7.148332118988037]], [[7.145591735839844]], [[5.6541337966918945]], [[7.456107139587402]], [[6.729430198669434]], [[7.303269863128662]], [[6.572075843811035]], [[6.8416900634765625]], [[6.775949954986572]], [[6.854101657867432]], [[7.444413661956787]], [[6.266705513000488]], [[6.909192085266113]], [[6.955999374389648]], [[7.005992889404297]], [[6.871369361877441]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]



class PrimitiveOp_df290d4dd1ff071aa1f7aea637728d4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 50, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1a7716f96566d28da910415e365a82d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df290d4dd1ff071aa1f7aea637728d4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38686e4211ddb2fa10fc19d11891df27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a52905ea4d1ceafc8927d22e04ee019
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_736687bcaf5dfaea1aae7336ee1fb060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.565829277038574]], [[7.230114459991455]], [[7.268362522125244]], [[7.61113977432251]], [[8.124086380004883]], [[7.080787658691406]], [[7.871699333190918]], [[7.5024800300598145]], [[8.012104034423828]], [[7.275782108306885]], [[6.873510360717773]], [[7.38699197769165]], [[7.346246719360352]], [[7.186011791229248]], [[6.85170841217041]], [[7.44446325302124]], [[7.454082012176514]], [[7.507258415222168]], [[6.450994968414307]], [[7.521205902099609]], [[7.782055377960205]], [[7.603641033172607]], [[7.582657814025879]], [[6.958940505981445]], [[7.8783135414123535]], [[7.65800666809082]], [[7.353710651397705]], [[6.795878887176514]], [[7.312654495239258]], [[8.331714630126953]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0ea400a98f56d30e91978e16bc7c651b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aaf3f9c9f43c1b89281b8ed63f65d42
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.318763256072998]], [[3.1680665016174316]], [[3.4006760120391846]], [[2.983863115310669]], [[3.640672206878662]], [[2.7548511028289795]], [[3.0417733192443848]], [[2.95063853263855]], [[3.4714832305908203]], [[2.9133124351501465]], [[3.6549854278564453]], [[2.5973129272460938]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_895fd63e4eff0d38cb6780d334436026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aaf3f9c9f43c1b89281b8ed63f65d42
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.7156894207000732]], [[3.0214409828186035]], [[2.8685126304626465]], [[2.6621015071868896]], [[3.023104429244995]], [[2.7299301624298096]], [[2.941941976547241]], [[2.5256056785583496]], [[3.179490327835083]], [[2.9562082290649414]], [[3.0781970024108887]], [[3.009678840637207]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05c7124fa1229f62e949199889e28077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ad8d2f145fdbb6389a6e940b3876077
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.140129089355469]], [[6.696193218231201]], [[7.465770721435547]], [[6.132399082183838]], [[6.829151153564453]], [[7.21022891998291]], [[5.931122303009033]], [[6.4561944007873535]], [[6.971693992614746]], [[6.791543960571289]], [[7.606695652008057]], [[7.160106658935547]], [[6.561000823974609]], [[7.264130115509033]], [[7.151678085327148]], [[6.097128868103027]], [[6.880153656005859]], [[6.365011692047119]], [[7.272219181060791]], [[7.540491580963135]], [[6.16954231262207]], [[6.60228967666626]], [[7.431386470794678]], [[7.168369770050049]], [[6.6712470054626465]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]



class PrimitiveOp_9eb87c108dfd0ea1413bcaee6ed90ca5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0d44a17065b8be59d3d34df1bf757fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9eb87c108dfd0ea1413bcaee6ed90ca5
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ce50b3fa1726b6e72affe2211901b20f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 312], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ccd7454861fdc5e926b48d6a977f8769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce50b3fa1726b6e72affe2211901b20f
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c012659453ddd863292dae97cf000833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d787b651e173fdd72b62c54eb2369a9
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc22259e7ac5ff82d98f75a31435dcc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_238208547838856b8d51479a53e69c6d
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e0f90de90bea669139bf22e944e6415f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c3fd8c5ee103972a3ea935b746da0ed0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.789164066314697]], [[5.643020153045654]], [[5.97690486907959]], [[5.5626630783081055]], [[4.481626033782959]], [[4.3739728927612305]], [[5.175326347351074]], [[5.458105087280273]], [[5.494755268096924]], [[5.029382228851318]], [[5.153017520904541]], [[6.021791458129883]], [[5.209958076477051]], [[4.5759429931640625]], [[5.057290554046631]], [[4.93481969833374]], [[5.839115619659424]], [[5.26939058303833]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]



class PrimitiveOp_b6e368c9ff81158ef8cbd6e1e50e8a3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 39], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1536a1c357d5dbc985b13604e631985b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6e368c9ff81158ef8cbd6e1e50e8a3e
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_f18a90c697362949c353e010cdf07ae5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7a2003f12699eb08fd6991bc7d6b322e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f18a90c697362949c353e010cdf07ae5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9634483456611633]], [[1.2220590114593506]], [[0.9322486519813538]], [[1.4995936155319214]], [[1.848082184791565]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]



class PrimitiveOp_7b04b99ae0d930a294a8dcfea28d0756(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f6e84206186dd7820095b7f72b35d9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b04b99ae0d930a294a8dcfea28d0756
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.699578046798706]], [[2.4012017250061035]], [[2.719417095184326]], [[1.8311710357666016]], [[2.7352590560913086]], [[2.572166681289673]], [[2.669515609741211]], [[2.982595443725586]], [[3.1514785289764404]], [[2.3121845722198486]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]



class PrimitiveOp_074b2732436b8fcd52cbf71478d86508(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bddffa8f3dd5d7394735432c88db8a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_074b2732436b8fcd52cbf71478d86508
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.552570819854736]], [[5.500812530517578]], [[5.3695969581604]], [[5.737652778625488]], [[5.323770999908447]], [[4.5695672035217285]], [[5.192540168762207]], [[5.187602519989014]], [[5.381386756896973]], [[4.964413642883301]], [[5.290442943572998]], [[5.411705493927002]], [[5.3682451248168945]], [[4.7196173667907715]], [[5.558773040771484]], [[5.463458061218262]], [[4.62952184677124]], [[5.318906307220459]], [[5.716233253479004]], [[4.891338348388672]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]



class PrimitiveOp_0ac3017314973ad892428c9f1d8ed87c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba0a64ac96dc54cc894e82ad77332fe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ac3017314973ad892428c9f1d8ed87c
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1a7716f96566d28da910415e365a82d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df290d4dd1ff071aa1f7aea637728d4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2b777266b2716abf22b892a797401db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e04c7dfa61283a44e2d457562974ff9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a46b4f0d5d188955d59c82278df6b70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e5829195b5969957bbc2b8cd268588e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19359c54ba9fa814761aadfec38e9057
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e79a8ac8e1db56f935e6f400640b05b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c026065d8c14e23899bdb4a05c680b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d831d43bfc1caca15a4ff9b85a72c0f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.646809101104736]], [[6.630312919616699]], [[6.695497035980225]], [[6.501897811889648]], [[6.6338982582092285]], [[5.826445579528809]], [[7.08401346206665]], [[6.037014007568359]], [[6.314833164215088]], [[6.280517578125]], [[7.346470832824707]], [[5.917470932006836]], [[6.874499320983887]], [[5.993903160095215]], [[6.453760147094727]], [[6.079671859741211]], [[6.251778602600098]], [[5.502902030944824]], [[6.202261924743652]], [[6.000838279724121]], [[5.825873374938965]], [[5.712315082550049]], [[6.406418800354004]], [[6.156716346740723]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0d7471684e65a142b61db767e0eb2fca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d787b651e173fdd72b62c54eb2369a9
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f0010e6a3d256760c081a95ecf4112da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db3a420cbd0dbb845b550b1a09e56109
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.244784355163574]], [[3.14597225189209]], [[3.1673083305358887]], [[2.5790815353393555]], [[2.7022438049316406]], [[2.790731430053711]], [[2.477316379547119]], [[3.020921468734741]], [[2.5250065326690674]], [[2.7123544216156006]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a53ea823f9279fe86c8a37efcbdafb08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d787b651e173fdd72b62c54eb2369a9
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aeeb0e7c45c34800b0132ebe1f2004b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_0a995de933c01e5e6a5ec0f2ce5a554b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_191bb5793d3356ed24222f2758fb36f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a995de933c01e5e6a5ec0f2ce5a554b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 50, 76], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_972b96c27bb2638b4b8a232ff0b89632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_238208547838856b8d51479a53e69c6d
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e5829195b5969957bbc2b8cd268588e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19359c54ba9fa814761aadfec38e9057
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c5a1197a4013dc5de677c4b359f13f1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.600645542144775]], [[6.025652885437012]], [[5.252595901489258]], [[5.656764030456543]], [[5.597095012664795]], [[4.575072765350342]], [[4.75056266784668]], [[4.224430561065674]], [[4.978477954864502]], [[5.510122776031494]], [[5.074936866760254]], [[4.85659646987915]], [[4.531800270080566]], [[4.506976127624512]], [[4.9718217849731445]], [[4.979997634887695]], [[5.516716003417969]], [[5.046294212341309]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]



class PrimitiveOp_ee3bc7956b0e79210390fb0751d164e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3f52e0a885313c9170848f5812fd0f26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee3bc7956b0e79210390fb0751d164e9
    def get_inputs(self):
        return [
            paddle.to_tensor([[7.328573703765869, 6.247287750244141, 7.139821529388428, 8.04222583770752, 7.969217777252197, 7.186982154846191, 6.653398036956787, 7.3734540939331055, 6.929617881774902, 8.31694221496582, 7.523541450500488, 7.806892395019531, 7.436128616333008, 7.457583904266357, 7.253500461578369, 7.969172477722168, 7.317387104034424, 7.51309871673584, 7.171206474304199, 6.792637825012207, 6.585378646850586, 7.3003926277160645, 7.75144624710083, 7.329493999481201, 7.001270294189453, 7.774947643280029, 6.7110700607299805, 7.38953161239624, 8.684221267700195, 7.437966823577881]], dtype='float32').reshape([1, 30]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_182d33db9ba31b1536ed2b4b293b153d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46c6c96cf66b0c7a3d16150e304f6ce0
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_182d33db9ba31b1536ed2b4b293b153d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46c6c96cf66b0c7a3d16150e304f6ce0
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_456e5d883195310ae232b35385bd8fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c97351a28248394bc67eb13bf250f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_edfa0eddfbb54839fea131842f168414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.774138450622559]], [[8.292790412902832]], [[8.968801498413086]], [[7.791714668273926]], [[8.903491973876953]], [[8.0988130569458]], [[8.791671752929688]], [[7.705585479736328]], [[8.631093978881836]], [[7.838891506195068]], [[8.784720420837402]], [[9.171486854553223]], [[8.685553550720215]], [[7.543893814086914]], [[8.089362144470215]], [[8.796666145324707]], [[8.934507369995117]], [[7.943831443786621]], [[7.65798282623291]], [[9.639039993286133]], [[8.797916412353516]], [[8.338150024414062]], [[7.518290042877197]], [[8.445304870605469]], [[8.131668090820312]], [[8.742501258850098]], [[9.182304382324219]], [[8.474281311035156]], [[8.359662055969238]], [[8.220466613769531]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4d1c4d9918bbfbda5380a6ddb7b1c096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccabd4bdc8c15e1883d6b58f2ffcf4ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.232663869857788]], [[1.2149169445037842]], [[1.141710638999939]], [[1.3593788146972656]], [[1.2661292552947998]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b22b3cb47a1cb51de13a8b50d82ea449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db3a420cbd0dbb845b550b1a09e56109
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8489837646484375]], [[2.9082000255584717]], [[3.118204116821289]], [[2.851863145828247]], [[2.812828540802002]], [[3.0019869804382324]], [[2.66688871383667]], [[2.7025177478790283]], [[2.774034023284912]], [[2.442160129547119]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9aa331c4829942de68b0dc35b3f87852(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5047832242eaf2936e08bceb2f12c5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.964051723480225]], [[5.199416637420654]], [[5.825793266296387]], [[4.516406536102295]], [[6.164638042449951]], [[4.838378429412842]], [[5.208074569702148]], [[5.449761390686035]], [[5.951899528503418]], [[5.441315174102783]], [[5.3666462898254395]], [[5.847661018371582]], [[5.8262505531311035]], [[5.1926093101501465]], [[4.903972625732422]], [[6.295505523681641]], [[5.192882537841797]], [[6.0003437995910645]], [[5.622760772705078]], [[5.101004600524902]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61d8c5843ee71a4304fc62712513d2c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5eef8ac6424fc66bad7653eaeda28b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f82cb8392f29bd015fe7ff80b5036f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a30ddc94d09c5f2cfeb153b79834946
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.237004280090332]], [[4.609907627105713]], [[4.416264533996582]], [[4.318653583526611]], [[4.779269695281982]], [[3.8180856704711914]], [[4.0465922355651855]], [[4.277736663818359]], [[4.53281831741333]], [[4.004705905914307]], [[4.262744903564453]], [[3.961000442504883]], [[4.129077434539795]], [[4.584085941314697]], [[4.1762285232543945]], [[3.648818254470825]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_558d75a863a543e1d813e0e31cebba1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c513ae7f47e626fe4801c31de815b10d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_c4935bab54bb92d8bc88e7b43ce7babb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ccba68fb4ce172eb5d8727319e0c701d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4935bab54bb92d8bc88e7b43ce7babb
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ceb207f5b609da299fe8b9d441516f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ceb207f5b609da299fe8b9d441516f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ceb207f5b609da299fe8b9d441516f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ceb207f5b609da299fe8b9d441516f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ceb207f5b609da299fe8b9d441516f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ceb207f5b609da299fe8b9d441516f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ceb207f5b609da299fe8b9d441516f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ceb207f5b609da299fe8b9d441516f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5e688cce609933924044c56c35e8f8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5e688cce609933924044c56c35e8f8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5e688cce609933924044c56c35e8f8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5e688cce609933924044c56c35e8f8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5e688cce609933924044c56c35e8f8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5e688cce609933924044c56c35e8f8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5e688cce609933924044c56c35e8f8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5e688cce609933924044c56c35e8f8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d51bcd2c6815f883990c100b05818fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d51bcd2c6815f883990c100b05818fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d51bcd2c6815f883990c100b05818fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d51bcd2c6815f883990c100b05818fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d51bcd2c6815f883990c100b05818fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d51bcd2c6815f883990c100b05818fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d51bcd2c6815f883990c100b05818fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d51bcd2c6815f883990c100b05818fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f901dd80ed806474c7409d5ce2a5618d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7c9e485d398baed71cdc1ede6777d74
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4bbff5adc6304be723ac83be663b66d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85ed9acc28e00af511e7803f56e3294e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_687052194472efb5643741db2c9d64d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20328261b99abcac38b3c71db66df47f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.137460708618164]], [[4.19693660736084]], [[4.408685207366943]], [[3.56331467628479]], [[4.196001052856445]], [[3.8622491359710693]], [[3.980560779571533]], [[2.498504161834717]], [[3.517117500305176]], [[3.7020151615142822]], [[3.7709763050079346]], [[3.425459146499634]], [[3.874528408050537]], [[3.964928150177002]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b42f47e569a1e74642ff454df62e5116(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5047832242eaf2936e08bceb2f12c5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.686917304992676]], [[5.096485614776611]], [[5.965027809143066]], [[4.96099853515625]], [[4.44478178024292]], [[5.442465782165527]], [[4.996183395385742]], [[4.863162040710449]], [[4.862886905670166]], [[4.750537395477295]], [[4.071479320526123]], [[4.924983501434326]], [[4.832633972167969]], [[5.044865131378174]], [[4.573680877685547]], [[4.4232940673828125]], [[5.002318859100342]], [[5.060869216918945]], [[4.030064105987549]], [[4.549410820007324]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_72b643e259a6c1f226af67d7f6b4e2c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2b777266b2716abf22b892a797401db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e04c7dfa61283a44e2d457562974ff9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_235068d5e2428b5b3986a299e714647e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.862788200378418]], [[8.0242338180542]], [[8.260517120361328]], [[7.583072662353516]], [[7.7904582023620605]], [[6.507443428039551]], [[6.787545204162598]], [[7.211781978607178]], [[7.58246374130249]], [[7.500139236450195]], [[7.678554058074951]], [[7.660146236419678]], [[8.634683609008789]], [[7.8989410400390625]], [[8.009509086608887]], [[7.43114709854126]], [[7.062746524810791]], [[8.060820579528809]], [[7.436216831207275]], [[7.444088459014893]], [[8.041678428649902]], [[7.679352760314941]], [[7.8064284324646]], [[6.900677680969238]], [[7.7797417640686035]], [[7.341211318969727]], [[7.735816955566406]], [[8.205156326293945]], [[8.522574424743652]], [[7.331345558166504]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_724d888912d9f3f997ccf2be87c6634c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_889d3db9c1718c4c87c2966a10bba689
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61d8c5843ee71a4304fc62712513d2c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5eef8ac6424fc66bad7653eaeda28b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ccba68fb4ce172eb5d8727319e0c701d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4935bab54bb92d8bc88e7b43ce7babb
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_28678929183a0bfdd2867fa0faebfd0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11e50181806e25ed31483467fd1c390b
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b26501b0fea91a38179e96b2b80df9bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6faba843a864104047205ddfb8d0bbd
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e47353c0d2496753accff26bd12d3045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e47353c0d2496753accff26bd12d3045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b26501b0fea91a38179e96b2b80df9bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6faba843a864104047205ddfb8d0bbd
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e47353c0d2496753accff26bd12d3045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e47353c0d2496753accff26bd12d3045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed9f7824365c6f3982180c82faacc4e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffaeb78900b871c336f9c645fe5ad069(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffaeb78900b871c336f9c645fe5ad069(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_688fb3cd4c1c63ee9ea76746fa942c41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3df121c9e9f0694de0b8b190f1f72db2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3df121c9e9f0694de0b8b190f1f72db2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_94a16bc93adf8f33118a47722ad15150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5886281400e7eb7ed9c1f84512e93d36
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_06b4cff6458152569a185fa357200ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_06b4cff6458152569a185fa357200ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_94a16bc93adf8f33118a47722ad15150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5886281400e7eb7ed9c1f84512e93d36
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_06b4cff6458152569a185fa357200ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_06b4cff6458152569a185fa357200ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8689792eb669c33ff65d82acf42902b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6ec24b63eeb454ad39e888443eede6ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6ec24b63eeb454ad39e888443eede6ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_01e8b2a2835a04e5ec6aa688dce55286(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_149466f6e1dabef6a9c6a61a8ee0a992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_149466f6e1dabef6a9c6a61a8ee0a992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_51ae22b0a10bd39597d6ee50bb70fad7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d91316752f88d173c9cea18bb0ff959
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1a7716f96566d28da910415e365a82d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df290d4dd1ff071aa1f7aea637728d4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0514de580e73ecfcf3b5d7ddaad2014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93e2d925a5b151abf25605c695763c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72fe2a48ee5468f7c5c2d180a930b506
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c53c56b301c568f249158f216dadf3c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.923415660858154]], [[6.35469388961792]], [[5.446557998657227]], [[5.3276166915893555]], [[5.289573669433594]], [[6.38268518447876]], [[5.428426742553711]], [[5.490380764007568]], [[5.250273704528809]], [[5.42781925201416]], [[6.073063850402832]], [[6.309932708740234]], [[5.747925758361816]], [[5.763967990875244]], [[6.289633274078369]], [[6.425298690795898]], [[5.383092880249023]], [[4.940002918243408]], [[6.08852481842041]], [[5.713768005371094]], [[6.511539459228516]], [[6.0695695877075195]], [[5.941235542297363]], [[5.920079708099365]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_59c3603938e41e2607c802dc299f3a12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ad8d2f145fdbb6389a6e940b3876077
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.5931715965271]], [[7.042733192443848]], [[6.315180778503418]], [[7.322271347045898]], [[6.81873083114624]], [[6.845092296600342]], [[6.102399826049805]], [[6.608209609985352]], [[6.2780046463012695]], [[6.651045322418213]], [[6.960600852966309]], [[6.798151016235352]], [[7.113742828369141]], [[6.831503391265869]], [[6.7376813888549805]], [[6.68527889251709]], [[7.164302349090576]], [[6.148285388946533]], [[6.805474758148193]], [[6.060007572174072]], [[7.482273578643799]], [[6.092843055725098]], [[6.139388561248779]], [[6.29178524017334]], [[6.123867034912109]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ce1349c8474d13e22bc3fc831e39c47b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aaf3f9c9f43c1b89281b8ed63f65d42
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.045076370239258]], [[3.3071534633636475]], [[3.5409646034240723]], [[3.446868896484375]], [[3.3328943252563477]], [[3.895918369293213]], [[3.0889813899993896]], [[3.3767125606536865]], [[3.9853267669677734]], [[3.4294793605804443]], [[3.3304035663604736]], [[3.6294891834259033]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e5829195b5969957bbc2b8cd268588e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19359c54ba9fa814761aadfec38e9057
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61d8c5843ee71a4304fc62712513d2c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5eef8ac6424fc66bad7653eaeda28b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2b777266b2716abf22b892a797401db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e04c7dfa61283a44e2d457562974ff9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a46b4f0d5d188955d59c82278df6b70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_db50e9c28683fa08f4de46e72eed11b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_16abc52ccd26a46dce50f589e2428f67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db50e9c28683fa08f4de46e72eed11b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_558d75a863a543e1d813e0e31cebba1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c513ae7f47e626fe4801c31de815b10d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5a7cffdbffd6a433b04316df2ef247e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61d8c5843ee71a4304fc62712513d2c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5eef8ac6424fc66bad7653eaeda28b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e5829195b5969957bbc2b8cd268588e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19359c54ba9fa814761aadfec38e9057
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2c6d16199fb6e87c3bef398cba326ac6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9b3a70689ea256f17c75416145d4a7da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c6d16199fb6e87c3bef398cba326ac6
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 10], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9b355389199be089d4df0b2f8d0cd653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[762.9212646484375]], [[684.9712524414062]], [[795.0789184570312]], [[795.4382934570312]], [[794.3326416015625]], [[756.835693359375]], [[721.7741088867188]], [[722.9525146484375]], [[716.1341552734375]], [[812.900146484375]], [[802.2077026367188]], [[720.6615600585938]], [[764.1770629882812]], [[735.3428344726562]], [[693.1209106445312]], [[782.7120971679688]], [[683.7904663085938]], [[684.3321533203125]], [[739.3828735351562]], [[746.4010009765625]], [[698.2523193359375]], [[759.6157836914062]], [[753.7282104492188]], [[750.2522583007812]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d560ec24c09f3b996570ddf28a083690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[89.96979522705078]], [[97.06975555419922]], [[94.78624725341797]], [[96.31605529785156]], [[101.97637176513672]], [[86.62770080566406]], [[100.61054992675781]], [[90.7819595336914]], [[92.42610931396484]], [[100.13390350341797]], [[93.68507385253906]], [[106.9056167602539]], [[104.99040985107422]], [[79.48321533203125]], [[89.82923126220703]], [[92.32855224609375]], [[99.09110260009766]], [[92.79380798339844]], [[95.33216094970703]], [[90.06105041503906]], [[92.29129791259766]], [[92.74037170410156]], [[101.51195526123047]], [[92.98509216308594]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a489e24f106e2086a4f6a8d6920d3fe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[31.849517822265625]], [[31.19537353515625]], [[31.82516860961914]], [[29.597890853881836]], [[28.871686935424805]], [[27.756689071655273]], [[29.93401527404785]], [[29.049955368041992]], [[28.514198303222656]], [[28.94681167602539]], [[29.82387924194336]], [[30.433561325073242]], [[28.050094604492188]], [[25.366117477416992]], [[26.298046112060547]], [[33.50059127807617]], [[29.867944717407227]], [[29.592456817626953]], [[29.20745277404785]], [[27.73162078857422]], [[27.946727752685547]], [[32.640018463134766]], [[28.33488655090332]], [[25.398555755615234]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0c4f8b4a1eb888dbfe9a920f2ff21667(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[19.86290740966797]], [[21.019996643066406]], [[24.71961784362793]], [[22.856386184692383]], [[23.035497665405273]], [[21.344566345214844]], [[22.04350471496582]], [[20.59987449645996]], [[23.119140625]], [[21.046899795532227]], [[21.647777557373047]], [[21.76042366027832]], [[19.37025260925293]], [[22.160438537597656]], [[22.544475555419922]], [[22.335384368896484]], [[24.94203758239746]], [[21.891090393066406]], [[21.417692184448242]], [[21.955320358276367]], [[20.658748626708984]], [[22.36775016784668]], [[21.59503173828125]], [[20.006811141967773]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e805dbe3a1bacfd7b86537694ed95b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8276864881ce911fc9b794b28454fd15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[28646.916015625]], [[33774.16015625]], [[28444.341796875]], [[37167.60546875]], [[30552.681640625]], [[36143.234375]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eb1668a07395a6ccb9c664da4aae017c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8276864881ce911fc9b794b28454fd15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[40890.3515625]], [[35828.65234375]], [[36393.65625]], [[30153.62890625]], [[40254.71484375]], [[34459.16796875]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_26534d1e3357fd575f35d6144f835808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8276864881ce911fc9b794b28454fd15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43695.44140625]], [[39566.20703125]], [[43151.87109375]], [[46809.8828125]], [[42969.6015625]], [[36620.71875]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_492b573d726ffb526eaf34289878ec2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8276864881ce911fc9b794b28454fd15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[39981.00390625]], [[48129.765625]], [[40868.40625]], [[47179.60546875]], [[42339.2890625]], [[37937.57421875]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4966d39265d512c979a31facf9a60b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_724d888912d9f3f997ccf2be87c6634c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_889d3db9c1718c4c87c2966a10bba689
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0d44a17065b8be59d3d34df1bf757fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9eb87c108dfd0ea1413bcaee6ed90ca5
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a3e3b46987adf2a39b2ac9900782aacf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_20cd8865fcb73a027202095157d57f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.756651401519775]], [[5.916136264801025]], [[5.633271217346191]], [[6.614544868469238]], [[6.991631031036377]], [[6.89631462097168]], [[6.508160591125488]], [[6.861876964569092]], [[7.014251232147217]], [[6.075519561767578]], [[6.874660015106201]], [[6.3834733963012695]], [[6.843541145324707]], [[6.827953338623047]], [[5.92738676071167]], [[6.641572952270508]], [[6.383920669555664]], [[6.474310398101807]], [[6.107621669769287]], [[6.669478416442871]], [[6.466883659362793]], [[6.878195762634277]], [[6.546784400939941]], [[6.064446926116943]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]



class PrimitiveOp_95e8d6e61f483b9084506df27e3f81b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5cdfa78bf832b2add94f79c634a9ff77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95e8d6e61f483b9084506df27e3f81b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 100, 152], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_a4f4df29f08a776807aaca3fcf70d6a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bb030f8827c8fd4081e5e5416a29f2b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4f4df29f08a776807aaca3fcf70d6a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_182d33db9ba31b1536ed2b4b293b153d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46c6c96cf66b0c7a3d16150e304f6ce0
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_092b65dbd396b4d1c62e42527ab61109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_22381fbd1daa84325ea8679549f1bef6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c24506c14f8ef98f715fcb795139d499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.to_tensor([[4.7128376960754395, 4.874957084655762, 4.827646255493164, 4.325068473815918, 4.148613929748535, 4.263113975524902, 4.267441749572754, 4.787745475769043, 5.16745138168335, 4.418758392333984, 5.105741500854492, 5.473527431488037, 5.001097202301025, 4.5380330085754395, 5.232659339904785, 5.4267425537109375, 6.2028679847717285, 5.070359230041504]], dtype='float32').reshape([1, 18]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f372128a87c139ffaa6ac301badbd15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.to_tensor([[7.115499496459961, 6.420863628387451, 6.43386173248291, 6.495194911956787, 6.782318592071533, 6.629547595977783, 6.720054626464844, 6.494166851043701, 7.013171672821045, 6.801267147064209, 6.6948089599609375, 6.927262306213379, 7.087639808654785, 7.325156211853027, 5.409924030303955, 7.67278528213501, 7.237374782562256, 6.851604461669922, 6.978597640991211, 7.248482704162598, 7.14051628112793, 7.549689769744873, 6.527603626251221]], dtype='float32').reshape([1, 23]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3401804b02a29cd50a5bcb4b53aaa1da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_446a456a3b619b11e7fb755b90848e51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_57d7e1cd55a10db6e90a4fad0163e48e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5267e2e559495404adee770c1be62abc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bb8fba8be6aec9eb71462bd27ea36540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bb8fba8be6aec9eb71462bd27ea36540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_879465da1871797ec9c6362b8dd1b4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_613cef06cf86cb8660999f143bdab2b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.932435989379883]], [[8.983295440673828]], [[7.800779342651367]], [[7.770794868469238]], [[7.657711982727051]], [[8.570398330688477]], [[7.985377788543701]], [[8.547874450683594]], [[6.691704750061035]], [[8.065608978271484]], [[8.797544479370117]], [[6.905008792877197]], [[7.936824798583984]], [[7.9261016845703125]], [[7.785417079925537]], [[7.818709850311279]], [[7.979026794433594]], [[8.503885269165039]], [[7.2908759117126465]], [[7.6919965744018555]], [[8.23678207397461]], [[8.094239234924316]], [[8.717683792114258]], [[7.794269561767578]], [[7.646255016326904]], [[8.356583595275879]], [[8.700400352478027]], [[7.808797359466553]], [[7.841263771057129]], [[8.376038551330566]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_464af458cccfa94e9dc8d0edcaa1beb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f835c166a465ca503332b1fe10d73fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f835c166a465ca503332b1fe10d73fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f835c166a465ca503332b1fe10d73fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f835c166a465ca503332b1fe10d73fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f835c166a465ca503332b1fe10d73fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f835c166a465ca503332b1fe10d73fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f835c166a465ca503332b1fe10d73fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f835c166a465ca503332b1fe10d73fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ba0dcdecdaa4d18a1ed19338851a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ba0dcdecdaa4d18a1ed19338851a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ba0dcdecdaa4d18a1ed19338851a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ba0dcdecdaa4d18a1ed19338851a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ba0dcdecdaa4d18a1ed19338851a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ba0dcdecdaa4d18a1ed19338851a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ba0dcdecdaa4d18a1ed19338851a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7ba0dcdecdaa4d18a1ed19338851a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46327ce353f73c04ccdb95db1c22f76f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46327ce353f73c04ccdb95db1c22f76f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46327ce353f73c04ccdb95db1c22f76f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46327ce353f73c04ccdb95db1c22f76f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46327ce353f73c04ccdb95db1c22f76f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46327ce353f73c04ccdb95db1c22f76f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46327ce353f73c04ccdb95db1c22f76f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46327ce353f73c04ccdb95db1c22f76f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f96bc62e6e16b65dbf1a66098109b82a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f96bc62e6e16b65dbf1a66098109b82a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f96bc62e6e16b65dbf1a66098109b82a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f96bc62e6e16b65dbf1a66098109b82a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f96bc62e6e16b65dbf1a66098109b82a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f96bc62e6e16b65dbf1a66098109b82a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f96bc62e6e16b65dbf1a66098109b82a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f96bc62e6e16b65dbf1a66098109b82a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e66645fb520c750261ae5db24f7d529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e66645fb520c750261ae5db24f7d529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e66645fb520c750261ae5db24f7d529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e66645fb520c750261ae5db24f7d529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e66645fb520c750261ae5db24f7d529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e66645fb520c750261ae5db24f7d529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e66645fb520c750261ae5db24f7d529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e66645fb520c750261ae5db24f7d529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f23f11a05508b040812451b916ee30d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.320115089416504]], [[7.914754390716553]], [[7.95361328125]], [[7.301033973693848]], [[7.661606311798096]], [[6.876945495605469]], [[7.888410568237305]], [[7.199883937835693]], [[7.128437042236328]], [[7.673472881317139]], [[6.833292007446289]], [[8.126873970031738]], [[7.337831974029541]], [[8.28213119506836]], [[7.123475551605225]], [[6.652795791625977]], [[7.81361722946167]], [[6.97962760925293]], [[6.899313449859619]], [[8.064953804016113]], [[7.767551898956299]], [[7.7723774909973145]], [[7.7174973487854]], [[7.75868034362793]], [[8.044079780578613]], [[8.732399940490723]], [[6.866262435913086]], [[7.858814716339111]], [[7.264653205871582]], [[8.07474136352539]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96b462f30c39c94a210b4774af293322(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fa665fdd4100186b31e205040581608d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.8689806461334229]], [[1.5793933868408203]], [[1.1897251605987549]], [[1.9041125774383545]], [[1.7028452157974243]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f6c972b489727b0f7da04afbc2c532fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.5980634689331055]], [[2.7655820846557617]], [[2.7791013717651367]], [[3.011228561401367]], [[3.1274821758270264]], [[2.1656382083892822]], [[3.099547863006592]], [[2.400660753250122]], [[3.081286907196045]], [[2.728254556655884]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_42d3f785a68f35ddaefc9270daf0c347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93fb83215cbed54131a6814f846cdb40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.704773426055908]], [[5.791843891143799]], [[5.963242053985596]], [[5.590764999389648]], [[6.544337272644043]], [[5.998619556427002]], [[6.506511211395264]], [[5.367368221282959]], [[6.426480770111084]], [[6.108403205871582]], [[5.482844829559326]], [[5.846292972564697]], [[6.339420795440674]], [[6.519515037536621]], [[6.135477066040039]], [[5.573600769042969]], [[5.396278381347656]], [[5.879087448120117]], [[5.871366500854492]], [[6.803126335144043]], [[6.488256931304932]], [[5.959376811981201]], [[6.003581523895264]], [[6.510748863220215]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_640dfc096a2dd3353d5e664045fb8bfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_726796571fb0bd926f858b5e99776c68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d1546bc82fe2eb799580686b3e84e4f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_41ebcca23620fce1a9f9533e6df5be75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.729099273681641]], [[5.868844985961914]], [[5.271246433258057]], [[5.6339335441589355]], [[5.561883926391602]], [[6.020602226257324]], [[5.928430557250977]], [[5.490959644317627]], [[6.217313289642334]], [[5.175939083099365]], [[5.130229949951172]], [[5.410799980163574]], [[5.388720989227295]], [[5.670409202575684]], [[4.803288459777832]], [[5.071345329284668]], [[4.968140125274658]], [[4.7747883796691895]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_42d3f785a68f35ddaefc9270daf0c347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_232a2e77cd3e5356eca1c5849ac64fbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.410586357116699]], [[6.070959091186523]], [[6.3691086769104]], [[6.467810153961182]], [[5.7446370124816895]], [[6.573790550231934]], [[6.578603267669678]], [[6.422765254974365]], [[6.163666725158691]], [[6.709517955780029]], [[6.100464820861816]], [[6.599005222320557]], [[6.100033283233643]], [[6.477661609649658]], [[5.917914390563965]], [[6.493032932281494]], [[5.760428428649902]], [[6.31825590133667]], [[6.541898727416992]], [[5.9510064125061035]], [[6.197717189788818]], [[6.384490013122559]], [[6.556829929351807]], [[6.843836784362793]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0ff1bb038b65a20fb32ac2296db3f953(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f023a7523dc66aaf5816fa4875f37c8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e7b62afee081d3486bf058bae1e44a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9010274410247803]], [[0.8791264295578003]], [[0.7772576808929443]], [[1.2044074535369873]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0ff1bb038b65a20fb32ac2296db3f953(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_761c1be698e51028abe4ee64988adea3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.2572855949401855]], [[2.820734739303589]], [[2.724828004837036]], [[2.7143571376800537]], [[2.5331223011016846]], [[2.0727102756500244]], [[3.1677427291870117]], [[2.7983739376068115]], [[2.8266916275024414]], [[2.6178479194641113]], [[2.9327988624572754]]]], dtype='float32').reshape([1, 11, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3401804b02a29cd50a5bcb4b53aaa1da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_42d3f785a68f35ddaefc9270daf0c347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47c034594326f8389515e0716c592668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b481d4c0c889b2714432d72df0ba415a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.317871570587158]], [[7.2048749923706055]], [[7.540952205657959]], [[7.160888195037842]], [[8.213842391967773]], [[7.61870002746582]], [[8.003351211547852]], [[7.220874786376953]], [[7.4271111488342285]], [[7.651928901672363]], [[7.9275431632995605]], [[8.04538345336914]], [[8.244603157043457]], [[7.6148905754089355]], [[7.728623867034912]], [[6.60276985168457]], [[8.527016639709473]], [[8.105579376220703]], [[7.484890460968018]], [[8.051274299621582]], [[7.621427536010742]], [[8.911153793334961]], [[7.644719123840332]], [[7.04080867767334]], [[7.401798248291016]], [[7.39521598815918]], [[7.999631881713867]], [[7.981103897094727]], [[7.7772536277771]], [[7.789159297943115]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_879465da1871797ec9c6362b8dd1b4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f0624971cbd4a881c7610f16101a9101(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0a9f68967862ba44e17d58f2f6ac06c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dcb361b85b6a4c0b0c7e6f0c7f9d94e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.879011154174805]], [[4.125312328338623]], [[4.7622294425964355]], [[3.8742642402648926]], [[3.6968116760253906]], [[4.492135047912598]], [[3.9997384548187256]], [[4.000436782836914]], [[4.1021928787231445]], [[4.1837239265441895]], [[4.110951900482178]], [[4.510783672332764]], [[4.736892223358154]], [[4.0170769691467285]], [[4.723196506500244]], [[4.257722854614258]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fb64d54ab6dbdad0f38aff15c92b1dd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8d10dfd8390172a6d33e401a50d8e008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_01f2dd3158c1d89d2366eeafbeab9a2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89baffe015f7216d4cdce9b5c9c10dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_879465da1871797ec9c6362b8dd1b4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e91b73c7d96c894da88047aa730444a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_31d2f46305c95cf7b097b001043f9d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47c034594326f8389515e0716c592668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cd3ec8094be9e3e7f1cca9693c3b3ba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.063333511352539]], [[7.867304801940918]], [[8.279768943786621]], [[8.106486320495605]], [[7.967275142669678]], [[8.184263229370117]], [[8.04330825805664]], [[9.142849922180176]], [[8.232620239257812]], [[8.164482116699219]], [[8.263772010803223]], [[8.041988372802734]], [[9.274855613708496]], [[7.256785869598389]], [[8.023280143737793]], [[8.690044403076172]], [[8.282241821289062]], [[7.736745357513428]], [[8.28187084197998]], [[8.18561840057373]], [[7.383940696716309]], [[8.074625968933105]], [[8.256736755371094]], [[8.356426239013672]], [[8.775242805480957]], [[7.776888847351074]], [[8.092998504638672]], [[8.468311309814453]], [[8.004794120788574]], [[8.935215950012207]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efa9795366e794b5f707ee2333924781(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_586ef4bd6caa9ffe584fd3d46d57e1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a893aeee02322ca8f5be7b68f520ed31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.039806365966797]], [[5.792285919189453]], [[5.926527500152588]], [[6.2489728927612305]], [[6.215531826019287]], [[6.358044624328613]], [[6.613975524902344]], [[6.986159801483154]], [[6.027462005615234]], [[5.791606426239014]], [[6.820392608642578]], [[6.442506313323975]], [[5.754520416259766]], [[6.488327980041504]], [[6.455815315246582]], [[6.31831169128418]], [[6.666366100311279]], [[6.836677551269531]], [[6.048222541809082]], [[6.6284098625183105]], [[6.555262088775635]], [[6.51648473739624]], [[5.54529333114624]], [[6.547698974609375]], [[7.419841289520264]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_42d3f785a68f35ddaefc9270daf0c347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2a45bfa5bf36752434cbbf0790ed4147(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_879465da1871797ec9c6362b8dd1b4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d14876fbacc008f768bef877e95bf35a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_da6afc03045d49075bb50fe89d743e85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_da6afc03045d49075bb50fe89d743e85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ff690b20520e7b747adf723f1f300324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e38b5168b78349770973007d1bb595b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c40ef0e605b88991a61528cd5984c6c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.280507564544678]], [[5.606911659240723]], [[5.503065586090088]], [[5.575466156005859]], [[5.39021110534668]], [[6.354246616363525]], [[6.104427814483643]], [[6.293274879455566]], [[6.576566696166992]], [[5.8172149658203125]], [[5.8143815994262695]], [[6.955115795135498]], [[6.240105628967285]], [[5.636475086212158]], [[6.0233564376831055]], [[5.5612969398498535]], [[5.591333389282227]], [[5.680702209472656]], [[5.9278082847595215]], [[5.499972820281982]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_879465da1871797ec9c6362b8dd1b4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efa9795366e794b5f707ee2333924781(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47c034594326f8389515e0716c592668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_879465da1871797ec9c6362b8dd1b4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f5fcc0d2e8b83dadfbb79f75463cb58e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.024784088134766]], [[5.615845680236816]], [[4.925999641418457]], [[5.293482780456543]], [[5.037164211273193]], [[4.709914207458496]], [[5.823171138763428]], [[5.3064680099487305]], [[5.286867618560791]], [[5.390621662139893]], [[5.206210136413574]], [[5.673350811004639]], [[5.507486343383789]], [[5.8226518630981445]], [[5.315316200256348]], [[5.093769550323486]], [[4.747065544128418]], [[5.24083137512207]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a152b3e389f83c25129f7fef7e6d16b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_092b65dbd396b4d1c62e42527ab61109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_74f91af3ca094b6bd1f221bb488d6364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_879465da1871797ec9c6362b8dd1b4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba28d2c0d2ac305cb330bcbab2bee283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1885f7df7c99c8b6a3abce19722b2be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed1ab301c7e5ffa98dd8bf76917d9c80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed1ab301c7e5ffa98dd8bf76917d9c80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1885f7df7c99c8b6a3abce19722b2be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed1ab301c7e5ffa98dd8bf76917d9c80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed1ab301c7e5ffa98dd8bf76917d9c80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a343f8c4dfa539fabe2d3f4e551b11a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b296e67c7a15989a0401e9a82248bf32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b296e67c7a15989a0401e9a82248bf32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6e3f22cabcaae2c2246eafd5d9f82cdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8be4483914f541e97131197a04587f6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8be4483914f541e97131197a04587f6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_880ae0f2b67ca60abd3392037be6c682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_990bb21634197b53726efbdb8898454c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_990bb21634197b53726efbdb8898454c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_880ae0f2b67ca60abd3392037be6c682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_990bb21634197b53726efbdb8898454c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_990bb21634197b53726efbdb8898454c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5018383c75c1f9f715ca35ac187a3d20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_75d0ec8403dc2514ed8b67917399f8fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_75d0ec8403dc2514ed8b67917399f8fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e7cff64261df91950649e72aedce25f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68179bbda00ae25bb61e917b6fed0803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68179bbda00ae25bb61e917b6fed0803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_613d3999537851ef53f7cbd0a60f47f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e38b5168b78349770973007d1bb595b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05068aca08a791c6282c908201c2cc65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.0720062255859375]], [[5.208503246307373]], [[5.06249475479126]], [[4.5323896408081055]], [[5.542492389678955]], [[5.169071197509766]], [[4.995449066162109]], [[4.422159671783447]], [[4.600342750549316]], [[4.8190999031066895]], [[4.795933246612549]], [[4.74271297454834]], [[4.916698932647705]], [[5.364006042480469]], [[5.094284534454346]], [[5.894742012023926]], [[4.947288513183594]], [[4.60883092880249]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_464af458cccfa94e9dc8d0edcaa1beb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3b2cb133ed9aa8ec9a1402c8805c89a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.289717674255371]], [[5.984675407409668]], [[6.481191635131836]], [[6.516063690185547]], [[6.890284061431885]], [[7.198267459869385]], [[5.973833084106445]], [[6.689905643463135]], [[6.024934768676758]], [[5.820860862731934]], [[6.1219377517700195]], [[6.105299472808838]], [[5.8296356201171875]], [[6.187517166137695]], [[6.363892555236816]], [[5.8717851638793945]], [[6.022918701171875]], [[6.813684463500977]], [[6.439299583435059]], [[6.676745891571045]], [[6.68774938583374]], [[6.478757858276367]], [[6.2004899978637695]], [[6.275086879730225]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e0e92ed3c15ab33ebe2736a118074401(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f0697398e1e3e64b924bbf5bd234268e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.7462079524993896]], [[4.334737300872803]], [[4.708061218261719]], [[3.785792112350464]], [[4.199547290802002]], [[3.8263018131256104]], [[4.310879707336426]], [[3.833042860031128]], [[3.332792282104492]], [[4.134519100189209]], [[4.635512351989746]], [[4.0650634765625]], [[4.113148212432861]], [[4.272996425628662]], [[3.92016863822937]], [[4.341777801513672]], [[3.8617467880249023]], [[4.640167236328125]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1e2211dbc8077db890c66d7c8ff8621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d51cbff9a60e39c1793c385646392ba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6193dd572ce4d20a860195b9484b8508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.40058708190918]], [[4.558745384216309]], [[5.096811294555664]], [[4.13717794418335]], [[4.5481414794921875]], [[4.9642510414123535]], [[4.2817182540893555]], [[4.441032886505127]], [[4.358351230621338]], [[3.9219157695770264]], [[4.180481433868408]], [[4.55796480178833]], [[4.259859085083008]], [[4.347965717315674]], [[4.469634532928467]], [[5.419482707977295]], [[4.937188148498535]], [[4.925285816192627]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1e2211dbc8077db890c66d7c8ff8621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b1b84b0409832ad89d473ce72cba4c30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_33373fb20dec1eda3b91e69b77f22a2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3bdff070ac598df08ac5bfa448fa0a2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a78536f22d78f63de7ffe63015750ed1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_713b381b0bfc4f54ff416c207586f86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_713b381b0bfc4f54ff416c207586f86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a78536f22d78f63de7ffe63015750ed1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_713b381b0bfc4f54ff416c207586f86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_713b381b0bfc4f54ff416c207586f86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_56b29588d4c17bca909026df84e57038(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b930e4879c4318d4c7fe7330ea9221b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b930e4879c4318d4c7fe7330ea9221b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ca8b152cb7a7e7508e3c4f4b73bf3d73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ac981de4b8dfac05d0d2abce52a8fdd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ac981de4b8dfac05d0d2abce52a8fdd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6a8cf1536dad269fdf08ba10cad4655d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_af0e594c7628ada57802c0427118cce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_af0e594c7628ada57802c0427118cce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6a8cf1536dad269fdf08ba10cad4655d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_af0e594c7628ada57802c0427118cce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_af0e594c7628ada57802c0427118cce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f7257c7df4d94908e2f283414a7ea6e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf3dfc409429dbf5976ee98cd5dc95e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf3dfc409429dbf5976ee98cd5dc95e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7bb6c31c46614653f7ee710542887353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f823f7ba36751fc32f7d551e6a7ee895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f823f7ba36751fc32f7d551e6a7ee895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2b07d9ac2f242b99f910788f50bae5ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ccf5a336767d84122d5ebb7e6708cb31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_31d2f46305c95cf7b097b001043f9d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7356e4cf4a8843361dc1d4304aa49bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55eec19f18de1200e2b32bbfe14b16f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f2601f63dead1c9a2c0a5fbba57bc056(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ff690b20520e7b747adf723f1f300324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7990a0c5f16903155a351e64f9b4e294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7990a0c5f16903155a351e64f9b4e294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_da883cdaf42311c466d1b5b460fefee0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_da883cdaf42311c466d1b5b460fefee0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8aef182d1155509105c4e64b1a01089d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8aef182d1155509105c4e64b1a01089d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8aef182d1155509105c4e64b1a01089d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0dcb5ba085782e2b5ae8736f3820782a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0dcb5ba085782e2b5ae8736f3820782a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0dcb5ba085782e2b5ae8736f3820782a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f41233d3d7aa34928d639f1a096edda0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f41233d3d7aa34928d639f1a096edda0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f41233d3d7aa34928d639f1a096edda0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34391d4e4f8d6ce36348e326bc84d9a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34391d4e4f8d6ce36348e326bc84d9a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4d05029828517a64710f5a5aa290c534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cfff910cc5302d4d5efeb9b90e846daf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e33d734094bb3a4be64418f674d806c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1c48b2ad7441b4fe514083d71befd893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d12cd267519a1a3e278745ec3bc5330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c77a00164171a8236ba259fcbb7ad3b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89678cb65773735651f55ab0e0fff433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed020ae26d2872204a8ef84f6d35817f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_879465da1871797ec9c6362b8dd1b4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea2e767e1b4503e8c333d41327d9e973(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_297f2a7018404b8e6d85d504c3485a7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78c41a3a3f2be954e260437f1e432417(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4419ed6dbc2ddde534f614d66170b06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5da4e68226f2ae9537aecba469f23f29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.744770050048828]], [[6.469620227813721]], [[4.977606296539307]], [[5.104647636413574]], [[5.340109825134277]], [[4.746143817901611]], [[5.46645975112915]], [[4.978821277618408]], [[4.684392929077148]], [[5.801762580871582]], [[5.27894926071167]], [[4.450350284576416]], [[5.514953136444092]], [[4.709138870239258]], [[4.926785469055176]], [[5.040286540985107]], [[5.034231662750244]], [[4.999880313873291]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e38b5168b78349770973007d1bb595b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8ef4fcbede536c985194725fa3c9a37d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3401804b02a29cd50a5bcb4b53aaa1da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_309bcbdb0caa60528e4802a61354e730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d35f702b02cabbbb13ebda376678b690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.15363883972168]], [[4.262726783752441]], [[4.043941974639893]], [[3.477867603302002]], [[3.7144417762756348]], [[3.9724888801574707]], [[4.637231826782227]], [[3.9909329414367676]], [[3.735192060470581]], [[3.4399991035461426]], [[4.4346394538879395]], [[4.366098880767822]], [[3.759702205657959]], [[4.257101535797119]], [[4.576810836791992]], [[3.867220401763916]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a125ab3260ce80153d9b949a877b2872(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47c034594326f8389515e0716c592668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a645776b597a26e01498bbe183da904a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.576985836029053]], [[4.620308876037598]], [[4.691135406494141]], [[4.501779556274414]], [[4.308021068572998]], [[4.544992446899414]], [[4.8089189529418945]], [[5.429499626159668]], [[4.519501686096191]], [[5.360790252685547]], [[4.243905067443848]], [[4.462162971496582]], [[4.71541166305542]], [[4.232010841369629]], [[4.425318717956543]], [[5.0960869789123535]], [[4.9816975593566895]], [[4.895888805389404]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5edb91bacb5a33b2dc30f82da17743cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0201356410980225]], [[1.2915623188018799]], [[0.6410092115402222]], [[1.0970127582550049]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5fe976d49df7abeeda3f8ba34644b404(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_309ba78f54f0b6414c680144777b2b08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dae118162b003bbfe32044e2a3a181df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dae118162b003bbfe32044e2a3a181df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_309ba78f54f0b6414c680144777b2b08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dae118162b003bbfe32044e2a3a181df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dae118162b003bbfe32044e2a3a181df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1015aa4401c089713afa8cb9756e1f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5bea4b62e72ec355e7b36404bd75638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5bea4b62e72ec355e7b36404bd75638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_43e59b5052dd516656bfdaa78308598f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7801d44ca0095b3e6161b22db6b465ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7801d44ca0095b3e6161b22db6b465ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c4e273e55aa578aedc5bb0e0cd6b6974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c479b777a317fda617482d930a73fff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c479b777a317fda617482d930a73fff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c4e273e55aa578aedc5bb0e0cd6b6974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c479b777a317fda617482d930a73fff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c479b777a317fda617482d930a73fff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d044510afbac988a587b29e3f3506f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_74025c82d095c3c41987c40e8f7af9ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_74025c82d095c3c41987c40e8f7af9ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c2ea6540a0fdfde08dff061ba6b098d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dcdd1314ded8dc52744122acea4bf802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dcdd1314ded8dc52744122acea4bf802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9b5cd3739909a438068995266dc5af48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e38b5168b78349770973007d1bb595b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d14876fbacc008f768bef877e95bf35a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_44abae1cba62ff965888cdd3dcc8075f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e431e684312e2fd7d42689b2089534d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32926fcd9fecccab88b4d097ddbc9b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7356e4cf4a8843361dc1d4304aa49bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e38b5168b78349770973007d1bb595b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47c034594326f8389515e0716c592668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1e2211dbc8077db890c66d7c8ff8621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_879465da1871797ec9c6362b8dd1b4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f098a00a13831beccf2b4f8e05abb4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.691684722900391]], [[5.826727390289307]], [[5.569563865661621]], [[4.746211528778076]], [[5.540277004241943]], [[5.252917766571045]], [[4.920151233673096]], [[4.923593044281006]], [[4.93858528137207]], [[5.285863399505615]], [[5.395374298095703]], [[5.400188446044922]], [[5.340855121612549]], [[5.122163772583008]], [[5.982247352600098]], [[6.027462005615234]], [[5.638346195220947]], [[5.619996070861816]], [[5.0743632316589355]], [[5.145237445831299]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99ec7f20d2cde5b4ed74fe3c6545f778(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_73c1468b1713bb418746040baa29fae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.3242204189300537]], [[2.5504298210144043]], [[3.132251262664795]], [[2.8135931491851807]], [[3.2420735359191895]], [[2.9025022983551025]], [[3.072528839111328]], [[2.5479371547698975]], [[2.4486899375915527]], [[3.069793939590454]], [[3.234339475631714]], [[3.066904067993164]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f2d253f37e71379371be72135ead154d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.2184977531433105]], [[5.2667107582092285]], [[5.434055328369141]], [[5.131888389587402]], [[4.453257083892822]], [[6.212052345275879]], [[5.52595853805542]], [[4.89768123626709]], [[4.918511867523193]], [[5.475644588470459]], [[5.82695198059082]], [[5.78203010559082]], [[5.754968643188477]], [[5.206923484802246]], [[5.637141704559326]], [[5.170568943023682]], [[5.467113494873047]], [[4.992288589477539]], [[4.318904876708984]], [[5.6925435066223145]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6832adf141c9a526686844216d8ef3f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.4854507446289062]], [[2.4385039806365967]], [[2.5216965675354004]], [[2.1648199558258057]], [[2.632795810699463]], [[2.3447458744049072]], [[3.06015944480896]], [[2.8932671546936035]], [[2.719749927520752]], [[2.519822835922241]], [[2.353712797164917]]]], dtype='float32').reshape([1, 11, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47c034594326f8389515e0716c592668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32926fcd9fecccab88b4d097ddbc9b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b1c86ecb68e75819f3a9f95e02d40639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6c69f3713db77eb90af6790b4298fcd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.3554539680480957]], [[3.217783212661743]], [[2.8873744010925293]], [[3.1918511390686035]], [[3.068450450897217]], [[2.8076624870300293]], [[3.5078229904174805]], [[3.385564088821411]], [[3.763841390609741]], [[3.3057873249053955]], [[3.5406668186187744]], [[3.7813518047332764]], [[3.6586873531341553]], [[3.3288607597351074]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aeafd0e0cfd32372fdd73d0c65940d86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_726796571fb0bd926f858b5e99776c68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3401804b02a29cd50a5bcb4b53aaa1da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea6392fe5f0249b5ac19b6182df64ae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.097700119018555]], [[5.456888675689697]], [[5.51584529876709]], [[5.612609386444092]], [[5.830142974853516]], [[5.613444805145264]], [[5.420537948608398]], [[6.005110263824463]], [[5.142417907714844]], [[5.411586284637451]], [[5.896563529968262]], [[4.884195327758789]], [[6.2467546463012695]], [[5.234903812408447]], [[5.787017822265625]], [[5.4528889656066895]], [[5.498476028442383]], [[5.138840675354004]], [[5.521658897399902]], [[5.879971981048584]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55f7f318239192a5e698930498a7850c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55f7f318239192a5e698930498a7850c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55f7f318239192a5e698930498a7850c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55f7f318239192a5e698930498a7850c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b8aca4ff981faa0f2d41b304612bac1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[31571.841796875]], [[35344.24609375]], [[36648.8671875]], [[35764.453125]], [[35387.73046875]], [[32963.13671875]]], [[[31131.6484375]], [[34854.49609375]], [[36155.8203125]], [[35275.0234375]], [[34907.05078125]], [[32515.74609375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a99c07c81391fd0200e7c03ce07ac452(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[36937.890625]], [[44298.43359375]], [[38864.83984375]], [[40726.38671875]], [[46830.0546875]], [[40810.7578125]]], [[[34079.921875]], [[40872.62890625]], [[35855.78515625]], [[37574.31640625]], [[43202.19140625]], [[37654.83984375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c79295fd914bd0d4829f573bef7d0a63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[41162.078125]], [[43691.71875]], [[48400.57421875]], [[33811.140625]], [[47499.08984375]], [[44477.12890625]]], [[[37595.48828125]], [[39901.8125]], [[44203.8828125]], [[30882.15234375]], [[43378.43359375]], [[40617.95703125]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7864b921b088c372e89860370ad7f3e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43057.671875]], [[50181.33203125]], [[45159.3359375]], [[42627.359375]], [[43513.625]], [[43262.671875]]], [[[39031.8515625]], [[45485.43359375]], [[40927.40625]], [[38637.046875]], [[39445.37109375]], [[39215.1484375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e05f9da72ceb520b50a48939f69b175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e05f9da72ceb520b50a48939f69b175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e05f9da72ceb520b50a48939f69b175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e05f9da72ceb520b50a48939f69b175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e05f9da72ceb520b50a48939f69b175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e05f9da72ceb520b50a48939f69b175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e05f9da72ceb520b50a48939f69b175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e05f9da72ceb520b50a48939f69b175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4feb74016a940aadd744c0fb789d4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4feb74016a940aadd744c0fb789d4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4feb74016a940aadd744c0fb789d4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4feb74016a940aadd744c0fb789d4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4feb74016a940aadd744c0fb789d4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4feb74016a940aadd744c0fb789d4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4feb74016a940aadd744c0fb789d4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4feb74016a940aadd744c0fb789d4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52693719c6145aa382af0595283d22c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52693719c6145aa382af0595283d22c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52693719c6145aa382af0595283d22c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52693719c6145aa382af0595283d22c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52693719c6145aa382af0595283d22c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52693719c6145aa382af0595283d22c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52693719c6145aa382af0595283d22c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52693719c6145aa382af0595283d22c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47c034594326f8389515e0716c592668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1e2211dbc8077db890c66d7c8ff8621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d936cd99250199c202003c514c5a69d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.533145904541016]], [[7.354318141937256]], [[7.447439670562744]], [[7.0797119140625]], [[7.624172687530518]], [[7.093131065368652]], [[6.802231311798096]], [[6.923765182495117]], [[7.4682817459106445]], [[7.509212970733643]], [[6.72806453704834]], [[7.412635326385498]], [[7.149740219116211]], [[7.231971740722656]], [[6.481606483459473]], [[8.51262378692627]], [[7.945773601531982]], [[7.963558673858643]], [[7.250692367553711]], [[6.296770095825195]], [[7.527583599090576]], [[7.976482391357422]], [[7.857499599456787]], [[7.265157699584961]], [[7.281493186950684]], [[6.802944660186768]], [[8.28808879852295]], [[6.954543113708496]], [[7.341640472412109]], [[7.064842224121094]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_631dd07976c13f1e6ef0f66d22536799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.124375343322754]], [[7.7985639572143555]], [[8.486802101135254]], [[8.122688293457031]], [[8.025605201721191]], [[7.954373359680176]], [[8.63216781616211]], [[8.88422679901123]], [[7.932723522186279]], [[9.44963264465332]], [[9.011527061462402]], [[8.29687213897705]], [[8.6183443069458]], [[8.34325885772705]], [[8.56541633605957]], [[7.821725845336914]], [[7.581669807434082]], [[8.553250312805176]], [[7.7252655029296875]], [[7.795560836791992]], [[8.779960632324219]], [[8.769819259643555]], [[8.215607643127441]], [[7.944150924682617]], [[7.799108028411865]], [[8.4075345993042]], [[8.656457901000977]], [[8.188000679016113]], [[8.324912071228027]], [[8.560117721557617]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5b587a6b5a79acfc31d910ed09b8cf67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_79fbe99eb5d54e702b8c2e7e982e43b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.451113224029541]], [[7.2594804763793945]], [[7.049568176269531]], [[7.04967737197876]], [[7.738953590393066]], [[6.293715476989746]], [[7.056695461273193]], [[6.882185459136963]], [[7.1396164894104]], [[6.536636829376221]], [[7.101452350616455]], [[6.9430952072143555]], [[6.751807689666748]], [[6.944225311279297]], [[7.148332118988037]], [[7.145591735839844]], [[5.6541337966918945]], [[7.456107139587402]], [[6.729430198669434]], [[7.303269863128662]], [[6.572075843811035]], [[6.8416900634765625]], [[6.775949954986572]], [[6.854101657867432]], [[7.444413661956787]], [[6.266705513000488]], [[6.909192085266113]], [[6.955999374389648]], [[7.005992889404297]], [[6.871369361877441]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_71d67eb974ef937e0f20713d0842d4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e38b5168b78349770973007d1bb595b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77e4e6a588e20db08cabdbc453501675(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.565829277038574]], [[7.230114459991455]], [[7.268362522125244]], [[7.61113977432251]], [[8.124086380004883]], [[7.080787658691406]], [[7.871699333190918]], [[7.5024800300598145]], [[8.012104034423828]], [[7.275782108306885]], [[6.873510360717773]], [[7.38699197769165]], [[7.346246719360352]], [[7.186011791229248]], [[6.85170841217041]], [[7.44446325302124]], [[7.454082012176514]], [[7.507258415222168]], [[6.450994968414307]], [[7.521205902099609]], [[7.782055377960205]], [[7.603641033172607]], [[7.582657814025879]], [[6.958940505981445]], [[7.8783135414123535]], [[7.65800666809082]], [[7.353710651397705]], [[6.795878887176514]], [[7.312654495239258]], [[8.331714630126953]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7a6d4ca8df2643c06ee6016741acca86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.318763256072998]], [[3.1680665016174316]], [[3.4006760120391846]], [[2.983863115310669]], [[3.640672206878662]], [[2.7548511028289795]], [[3.0417733192443848]], [[2.95063853263855]], [[3.4714832305908203]], [[2.9133124351501465]], [[3.6549854278564453]], [[2.5973129272460938]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_af0e25d28f38183f1cac7253b5556c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.7156894207000732]], [[3.0214409828186035]], [[2.8685126304626465]], [[2.6621015071868896]], [[3.023104429244995]], [[2.7299301624298096]], [[2.941941976547241]], [[2.5256056785583496]], [[3.179490327835083]], [[2.9562082290649414]], [[3.0781970024108887]], [[3.009678840637207]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_537d4c4e27ba8bc19f27633728e6996d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.140129089355469]], [[6.696193218231201]], [[7.465770721435547]], [[6.132399082183838]], [[6.829151153564453]], [[7.21022891998291]], [[5.931122303009033]], [[6.4561944007873535]], [[6.971693992614746]], [[6.791543960571289]], [[7.606695652008057]], [[7.160106658935547]], [[6.561000823974609]], [[7.264130115509033]], [[7.151678085327148]], [[6.097128868103027]], [[6.880153656005859]], [[6.365011692047119]], [[7.272219181060791]], [[7.540491580963135]], [[6.16954231262207]], [[6.60228967666626]], [[7.431386470794678]], [[7.168369770050049]], [[6.6712470054626465]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b47adf61accad47610565706c0712cbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eca32c4a11018f1951438540b4e8fdb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f1b45f67fe34021a70d6a79a51075505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b5b229243a9e9f3b345f7858600d2cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8bbf899625f58e211ff73edb68db3f43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_90887d59bbf86d72080a0f63738c78e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.789164066314697]], [[5.643020153045654]], [[5.97690486907959]], [[5.5626630783081055]], [[4.481626033782959]], [[4.3739728927612305]], [[5.175326347351074]], [[5.458105087280273]], [[5.494755268096924]], [[5.029382228851318]], [[5.153017520904541]], [[6.021791458129883]], [[5.209958076477051]], [[4.5759429931640625]], [[5.057290554046631]], [[4.93481969833374]], [[5.839115619659424]], [[5.26939058303833]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4e3e5bd3b7aa5ec9b9d4430d8378b714(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_380d535981af71439e2424679729e4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9634483456611633]], [[1.2220590114593506]], [[0.9322486519813538]], [[1.4995936155319214]], [[1.848082184791565]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e619a4e6a1d02a9d686e6f6813971b50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.699578046798706]], [[2.4012017250061035]], [[2.719417095184326]], [[1.8311710357666016]], [[2.7352590560913086]], [[2.572166681289673]], [[2.669515609741211]], [[2.982595443725586]], [[3.1514785289764404]], [[2.3121845722198486]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_17d47c171088eb408cc557c6b32d6fcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.552570819854736]], [[5.500812530517578]], [[5.3695969581604]], [[5.737652778625488]], [[5.323770999908447]], [[4.5695672035217285]], [[5.192540168762207]], [[5.187602519989014]], [[5.381386756896973]], [[4.964413642883301]], [[5.290442943572998]], [[5.411705493927002]], [[5.3682451248168945]], [[4.7196173667907715]], [[5.558773040771484]], [[5.463458061218262]], [[4.62952184677124]], [[5.318906307220459]], [[5.716233253479004]], [[4.891338348388672]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3401804b02a29cd50a5bcb4b53aaa1da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_71d67eb974ef937e0f20713d0842d4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efa9795366e794b5f707ee2333924781(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f96bc62e6e16b65dbf1a66098109b82a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47c034594326f8389515e0716c592668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_586ef4bd6caa9ffe584fd3d46d57e1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_865ded7fca0f4c25d2e4fa427c96c667(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.646809101104736]], [[6.630312919616699]], [[6.695497035980225]], [[6.501897811889648]], [[6.6338982582092285]], [[5.826445579528809]], [[7.08401346206665]], [[6.037014007568359]], [[6.314833164215088]], [[6.280517578125]], [[7.346470832824707]], [[5.917470932006836]], [[6.874499320983887]], [[5.993903160095215]], [[6.453760147094727]], [[6.079671859741211]], [[6.251778602600098]], [[5.502902030944824]], [[6.202261924743652]], [[6.000838279724121]], [[5.825873374938965]], [[5.712315082550049]], [[6.406418800354004]], [[6.156716346740723]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2041013682eaded7ecacb301e2e58a3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_86cb81f808d331399e4297750d0bdd88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.244784355163574]], [[3.14597225189209]], [[3.1673083305358887]], [[2.5790815353393555]], [[2.7022438049316406]], [[2.790731430053711]], [[2.477316379547119]], [[3.020921468734741]], [[2.5250065326690674]], [[2.7123544216156006]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15d54eb686939dd6ee056bac3ec425ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3ab070c6f1ad19b24d4dab303ab2db11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_207ced3fb8aef9175b31f81beb2a8256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 50, 76], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07d2a7b2504ddc0eacb4766e4ce98dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47c034594326f8389515e0716c592668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0844231dbac505ab3fbe1252ec40d7c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.600645542144775]], [[6.025652885437012]], [[5.252595901489258]], [[5.656764030456543]], [[5.597095012664795]], [[4.575072765350342]], [[4.75056266784668]], [[4.224430561065674]], [[4.978477954864502]], [[5.510122776031494]], [[5.074936866760254]], [[4.85659646987915]], [[4.531800270080566]], [[4.506976127624512]], [[4.9718217849731445]], [[4.979997634887695]], [[5.516716003417969]], [[5.046294212341309]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f406d321a121f77faead38ba532a9fd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.to_tensor([[7.328573703765869, 6.247287750244141, 7.139821529388428, 8.04222583770752, 7.969217777252197, 7.186982154846191, 6.653398036956787, 7.3734540939331055, 6.929617881774902, 8.31694221496582, 7.523541450500488, 7.806892395019531, 7.436128616333008, 7.457583904266357, 7.253500461578369, 7.969172477722168, 7.317387104034424, 7.51309871673584, 7.171206474304199, 6.792637825012207, 6.585378646850586, 7.3003926277160645, 7.75144624710083, 7.329493999481201, 7.001270294189453, 7.774947643280029, 6.7110700607299805, 7.38953161239624, 8.684221267700195, 7.437966823577881]], dtype='float32').reshape([1, 30]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_879465da1871797ec9c6362b8dd1b4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_879465da1871797ec9c6362b8dd1b4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e431e684312e2fd7d42689b2089534d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_daed6ee8e2ba72b102c12b3179dbbb08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.774138450622559]], [[8.292790412902832]], [[8.968801498413086]], [[7.791714668273926]], [[8.903491973876953]], [[8.0988130569458]], [[8.791671752929688]], [[7.705585479736328]], [[8.631093978881836]], [[7.838891506195068]], [[8.784720420837402]], [[9.171486854553223]], [[8.685553550720215]], [[7.543893814086914]], [[8.089362144470215]], [[8.796666145324707]], [[8.934507369995117]], [[7.943831443786621]], [[7.65798282623291]], [[9.639039993286133]], [[8.797916412353516]], [[8.338150024414062]], [[7.518290042877197]], [[8.445304870605469]], [[8.131668090820312]], [[8.742501258850098]], [[9.182304382324219]], [[8.474281311035156]], [[8.359662055969238]], [[8.220466613769531]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d30fe9971fbe9ef2336a97616f3fa73b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.232663869857788]], [[1.2149169445037842]], [[1.141710638999939]], [[1.3593788146972656]], [[1.2661292552947998]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0a9e39add97a7ef4cbe1d380dff938c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8489837646484375]], [[2.9082000255584717]], [[3.118204116821289]], [[2.851863145828247]], [[2.812828540802002]], [[3.0019869804382324]], [[2.66688871383667]], [[2.7025177478790283]], [[2.774034023284912]], [[2.442160129547119]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b305dff2468ade08674840867a467e03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.964051723480225]], [[5.199416637420654]], [[5.825793266296387]], [[4.516406536102295]], [[6.164638042449951]], [[4.838378429412842]], [[5.208074569702148]], [[5.449761390686035]], [[5.951899528503418]], [[5.441315174102783]], [[5.3666462898254395]], [[5.847661018371582]], [[5.8262505531311035]], [[5.1926093101501465]], [[4.903972625732422]], [[6.295505523681641]], [[5.192882537841797]], [[6.0003437995910645]], [[5.622760772705078]], [[5.101004600524902]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3401804b02a29cd50a5bcb4b53aaa1da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e10b08104b4b697e1fae708f53f709e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.237004280090332]], [[4.609907627105713]], [[4.416264533996582]], [[4.318653583526611]], [[4.779269695281982]], [[3.8180856704711914]], [[4.0465922355651855]], [[4.277736663818359]], [[4.53281831741333]], [[4.004705905914307]], [[4.262744903564453]], [[3.961000442504883]], [[4.129077434539795]], [[4.584085941314697]], [[4.1762285232543945]], [[3.648818254470825]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_31d2f46305c95cf7b097b001043f9d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05bbbedf805471186bd46f14cb402376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1c8322b540a576992822cdf8c9bbec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1c8322b540a576992822cdf8c9bbec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1c8322b540a576992822cdf8c9bbec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1c8322b540a576992822cdf8c9bbec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1c8322b540a576992822cdf8c9bbec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1c8322b540a576992822cdf8c9bbec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1c8322b540a576992822cdf8c9bbec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1c8322b540a576992822cdf8c9bbec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9dfcf9e6ae0639a3afa602e2f7fc88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9dfcf9e6ae0639a3afa602e2f7fc88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9dfcf9e6ae0639a3afa602e2f7fc88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9dfcf9e6ae0639a3afa602e2f7fc88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9dfcf9e6ae0639a3afa602e2f7fc88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9dfcf9e6ae0639a3afa602e2f7fc88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9dfcf9e6ae0639a3afa602e2f7fc88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9dfcf9e6ae0639a3afa602e2f7fc88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07338898e9a2394aa61e1cdcf2f8a24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07338898e9a2394aa61e1cdcf2f8a24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07338898e9a2394aa61e1cdcf2f8a24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07338898e9a2394aa61e1cdcf2f8a24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07338898e9a2394aa61e1cdcf2f8a24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07338898e9a2394aa61e1cdcf2f8a24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07338898e9a2394aa61e1cdcf2f8a24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07338898e9a2394aa61e1cdcf2f8a24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99ec7f20d2cde5b4ed74fe3c6545f778(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aeafd0e0cfd32372fdd73d0c65940d86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_da7ae091f727bb9af00e4ffdc3bdc529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.137460708618164]], [[4.19693660736084]], [[4.408685207366943]], [[3.56331467628479]], [[4.196001052856445]], [[3.8622491359710693]], [[3.980560779571533]], [[2.498504161834717]], [[3.517117500305176]], [[3.7020151615142822]], [[3.7709763050079346]], [[3.425459146499634]], [[3.874528408050537]], [[3.964928150177002]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ca35fa94d9fe231c9dd986e08cf72a5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.686917304992676]], [[5.096485614776611]], [[5.965027809143066]], [[4.96099853515625]], [[4.44478178024292]], [[5.442465782165527]], [[4.996183395385742]], [[4.863162040710449]], [[4.862886905670166]], [[4.750537395477295]], [[4.071479320526123]], [[4.924983501434326]], [[4.832633972167969]], [[5.044865131378174]], [[4.573680877685547]], [[4.4232940673828125]], [[5.002318859100342]], [[5.060869216918945]], [[4.030064105987549]], [[4.549410820007324]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cbaa91c541da991cc1283089be6abbcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efa9795366e794b5f707ee2333924781(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f2aabb543683fa5ac69060c89c6951c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.862788200378418]], [[8.0242338180542]], [[8.260517120361328]], [[7.583072662353516]], [[7.7904582023620605]], [[6.507443428039551]], [[6.787545204162598]], [[7.211781978607178]], [[7.58246374130249]], [[7.500139236450195]], [[7.678554058074951]], [[7.660146236419678]], [[8.634683609008789]], [[7.8989410400390625]], [[8.009509086608887]], [[7.43114709854126]], [[7.062746524810791]], [[8.060820579528809]], [[7.436216831207275]], [[7.444088459014893]], [[8.041678428649902]], [[7.679352760314941]], [[7.8064284324646]], [[6.900677680969238]], [[7.7797417640686035]], [[7.341211318969727]], [[7.735816955566406]], [[8.205156326293945]], [[8.522574424743652]], [[7.331345558166504]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_42d3f785a68f35ddaefc9270daf0c347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3401804b02a29cd50a5bcb4b53aaa1da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05bbbedf805471186bd46f14cb402376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_90b2665146e8acddcc311ab2513215e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5b25216e947d95e442ae7a675d642f37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b3b6953c815bb0efd817664f0381585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b3b6953c815bb0efd817664f0381585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5b25216e947d95e442ae7a675d642f37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b3b6953c815bb0efd817664f0381585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b3b6953c815bb0efd817664f0381585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77bffecbd6c38445f94e51244b00e5a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93c3fd2bd4824ccd9c65719d06eea1c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93c3fd2bd4824ccd9c65719d06eea1c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d080da28eebcf475c3447482f55d8f97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_327aeff3e39beb3bbe908f7425094dab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_327aeff3e39beb3bbe908f7425094dab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99343f3221b77313058352e27c6eeeb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f5f676f44960e31eacd927755dec29b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f5f676f44960e31eacd927755dec29b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99343f3221b77313058352e27c6eeeb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f5f676f44960e31eacd927755dec29b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f5f676f44960e31eacd927755dec29b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df27faebd82272f6979f0cf6d0d6990b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe78f8fb8d0628309513a3b18f0a4449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe78f8fb8d0628309513a3b18f0a4449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba4f9ca46f6928483d4fe04b5f969974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_02eb531dd27598c3cb929e534053d91b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_02eb531dd27598c3cb929e534053d91b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_117a6965472cf9cdf2d8ee92fcc11842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_71d67eb974ef937e0f20713d0842d4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7037d3650459f14dbe4accfeabe669e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32926fcd9fecccab88b4d097ddbc9b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b36cef135ab81c5d12979bdc25b563e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.923415660858154]], [[6.35469388961792]], [[5.446557998657227]], [[5.3276166915893555]], [[5.289573669433594]], [[6.38268518447876]], [[5.428426742553711]], [[5.490380764007568]], [[5.250273704528809]], [[5.42781925201416]], [[6.073063850402832]], [[6.309932708740234]], [[5.747925758361816]], [[5.763967990875244]], [[6.289633274078369]], [[6.425298690795898]], [[5.383092880249023]], [[4.940002918243408]], [[6.08852481842041]], [[5.713768005371094]], [[6.511539459228516]], [[6.0695695877075195]], [[5.941235542297363]], [[5.920079708099365]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f4b58550d1b580ed62e3c3431ee549d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.5931715965271]], [[7.042733192443848]], [[6.315180778503418]], [[7.322271347045898]], [[6.81873083114624]], [[6.845092296600342]], [[6.102399826049805]], [[6.608209609985352]], [[6.2780046463012695]], [[6.651045322418213]], [[6.960600852966309]], [[6.798151016235352]], [[7.113742828369141]], [[6.831503391265869]], [[6.7376813888549805]], [[6.68527889251709]], [[7.164302349090576]], [[6.148285388946533]], [[6.805474758148193]], [[6.060007572174072]], [[7.482273578643799]], [[6.092843055725098]], [[6.139388561248779]], [[6.29178524017334]], [[6.123867034912109]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_097030b827846bd9894a3a808f26015b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.045076370239258]], [[3.3071534633636475]], [[3.5409646034240723]], [[3.446868896484375]], [[3.3328943252563477]], [[3.895918369293213]], [[3.0889813899993896]], [[3.3767125606536865]], [[3.9853267669677734]], [[3.4294793605804443]], [[3.3304035663604736]], [[3.6294891834259033]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47c034594326f8389515e0716c592668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3401804b02a29cd50a5bcb4b53aaa1da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efa9795366e794b5f707ee2333924781(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f96bc62e6e16b65dbf1a66098109b82a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ce8267680df7a42e6226644a4b6a24da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_31d2f46305c95cf7b097b001043f9d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_00b3e905cc899e8521f5b9bd640efed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3401804b02a29cd50a5bcb4b53aaa1da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47c034594326f8389515e0716c592668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5d106b0689815e7f536b6c24a0dfff88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 10], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99332153086f9b1620f1409e90c18578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[762.9212646484375]], [[684.9712524414062]], [[795.0789184570312]], [[795.4382934570312]], [[794.3326416015625]], [[756.835693359375]], [[721.7741088867188]], [[722.9525146484375]], [[716.1341552734375]], [[812.900146484375]], [[802.2077026367188]], [[720.6615600585938]], [[764.1770629882812]], [[735.3428344726562]], [[693.1209106445312]], [[782.7120971679688]], [[683.7904663085938]], [[684.3321533203125]], [[739.3828735351562]], [[746.4010009765625]], [[698.2523193359375]], [[759.6157836914062]], [[753.7282104492188]], [[750.2522583007812]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e31950cc6b26dc75d637f95f5a64a90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[89.96979522705078]], [[97.06975555419922]], [[94.78624725341797]], [[96.31605529785156]], [[101.97637176513672]], [[86.62770080566406]], [[100.61054992675781]], [[90.7819595336914]], [[92.42610931396484]], [[100.13390350341797]], [[93.68507385253906]], [[106.9056167602539]], [[104.99040985107422]], [[79.48321533203125]], [[89.82923126220703]], [[92.32855224609375]], [[99.09110260009766]], [[92.79380798339844]], [[95.33216094970703]], [[90.06105041503906]], [[92.29129791259766]], [[92.74037170410156]], [[101.51195526123047]], [[92.98509216308594]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4cb1ac897eb294441382a12cfdab1d34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[31.849517822265625]], [[31.19537353515625]], [[31.82516860961914]], [[29.597890853881836]], [[28.871686935424805]], [[27.756689071655273]], [[29.93401527404785]], [[29.049955368041992]], [[28.514198303222656]], [[28.94681167602539]], [[29.82387924194336]], [[30.433561325073242]], [[28.050094604492188]], [[25.366117477416992]], [[26.298046112060547]], [[33.50059127807617]], [[29.867944717407227]], [[29.592456817626953]], [[29.20745277404785]], [[27.73162078857422]], [[27.946727752685547]], [[32.640018463134766]], [[28.33488655090332]], [[25.398555755615234]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_920db621dcf1b0baf01fd0a2d0f3057b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[19.86290740966797]], [[21.019996643066406]], [[24.71961784362793]], [[22.856386184692383]], [[23.035497665405273]], [[21.344566345214844]], [[22.04350471496582]], [[20.59987449645996]], [[23.119140625]], [[21.046899795532227]], [[21.647777557373047]], [[21.76042366027832]], [[19.37025260925293]], [[22.160438537597656]], [[22.544475555419922]], [[22.335384368896484]], [[24.94203758239746]], [[21.891090393066406]], [[21.417692184448242]], [[21.955320358276367]], [[20.658748626708984]], [[22.36775016784668]], [[21.59503173828125]], [[20.006811141967773]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8a342b267627065af41be1117292df21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[28646.916015625]], [[33774.16015625]], [[28444.341796875]], [[37167.60546875]], [[30552.681640625]], [[36143.234375]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40b57f3a5499a9e8c15105332653b2ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[40890.3515625]], [[35828.65234375]], [[36393.65625]], [[30153.62890625]], [[40254.71484375]], [[34459.16796875]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b91155ff0961abe8db8ed468d561473e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43695.44140625]], [[39566.20703125]], [[43151.87109375]], [[46809.8828125]], [[42969.6015625]], [[36620.71875]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07ce317316465cd6e162618919769815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[39981.00390625]], [[48129.765625]], [[40868.40625]], [[47179.60546875]], [[42339.2890625]], [[37937.57421875]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba0c4f481c77bb3567197468e4202dd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_42d3f785a68f35ddaefc9270daf0c347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b47adf61accad47610565706c0712cbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b6cd80f75f7eafb0ae48846efa065a85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3704a188aedfc949723fb3cdf1e96984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.756651401519775]], [[5.916136264801025]], [[5.633271217346191]], [[6.614544868469238]], [[6.991631031036377]], [[6.89631462097168]], [[6.508160591125488]], [[6.861876964569092]], [[7.014251232147217]], [[6.075519561767578]], [[6.874660015106201]], [[6.3834733963012695]], [[6.843541145324707]], [[6.827953338623047]], [[5.92738676071167]], [[6.641572952270508]], [[6.383920669555664]], [[6.474310398101807]], [[6.107621669769287]], [[6.669478416442871]], [[6.466883659362793]], [[6.878195762634277]], [[6.546784400939941]], [[6.064446926116943]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_59862ab7f1c98eb3ee597018396bad7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 100, 152], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_14cd5d8161c3c4181f532187e10dd36a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_879465da1871797ec9c6362b8dd1b4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()