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
class PrimitiveOp_158a36528405f0ad9565777f4bcaeb43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.pow(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_929060b89bb6e3d2ac2f93b9f5d55fca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_158a36528405f0ad9565777f4bcaeb43
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b09c5f049e55577da7ed58e5b8f51a0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_158a36528405f0ad9565777f4bcaeb43
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_af1b687286294a12e674b666c4d58685(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.pow(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32f2d53585c7ad97ea7caf93c2964a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af1b687286294a12e674b666c4d58685
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_d8f5130c60f5f754b2570eb2cfae8317(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.pow(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1a3af0299683e8608063185208b89f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8f5130c60f5f754b2570eb2cfae8317
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.47760626673698425], [0.44666361808776855], [0.4991475045681], [0.3156026005744934], [0.29203882813453674], [0.4573269784450531]]], dtype='float32').reshape([1, 6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32f2d53585c7ad97ea7caf93c2964a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af1b687286294a12e674b666c4d58685
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b4e663e188dee53c2361cc945826fb77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8f5130c60f5f754b2570eb2cfae8317
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.15016944706439972], [0.14966975152492523], [0.0473039448261261], [0.27084267139434814], [0.40264034271240234], [0.18267758190631866]]], dtype='float32').reshape([1, 6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e9caa6c1a3220654e5060c4235967ae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_158a36528405f0ad9565777f4bcaeb43
    def get_inputs(self):
        return [
            paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4e890e99ed08a485529a90358cc04ecd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.pow(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_847719940f9f799487f2fa919fc55d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e890e99ed08a485529a90358cc04ecd
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba7941cbc9230dde5ef4efeb2b3f21e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af1b687286294a12e674b666c4d58685
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_59dfbde6eb6d8c82e1fd315788cfb773(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e890e99ed08a485529a90358cc04ecd
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c07be7ace52e63b7f5595b91f9e560da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_158a36528405f0ad9565777f4bcaeb43
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.pow(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3c97f7d329c1958ab1750e2c1cabdaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3c97f7d329c1958ab1750e2c1cabdaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4b77288f5f4247b0b73b95efaaf1f811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4b77288f5f4247b0b73b95efaaf1f811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_808ff10ef9f3ec30a5f924d93cdccb5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_808ff10ef9f3ec30a5f924d93cdccb5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_197d9a6b4a39f94e8be99415bfeb8b4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_197d9a6b4a39f94e8be99415bfeb8b4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f9f56610d9020904a05d2580f1f3ffdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f9f56610d9020904a05d2580f1f3ffdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e1baca7b536edd20c6ff2034426ea9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e1baca7b536edd20c6ff2034426ea9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_08cb44b7f3bd70ccb166f0d7776cb43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_08cb44b7f3bd70ccb166f0d7776cb43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c44eafc82106fb55a87674a701bc400f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c44eafc82106fb55a87674a701bc400f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f9f56610d9020904a05d2580f1f3ffdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f9f56610d9020904a05d2580f1f3ffdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e1baca7b536edd20c6ff2034426ea9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e1baca7b536edd20c6ff2034426ea9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_08cb44b7f3bd70ccb166f0d7776cb43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_08cb44b7f3bd70ccb166f0d7776cb43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c44eafc82106fb55a87674a701bc400f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c44eafc82106fb55a87674a701bc400f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9f9201d954dee2ec3d3ba5003e5d228d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_158a36528405f0ad9565777f4bcaeb43
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_393cc27045d7ba1f3219534ef030701a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e890e99ed08a485529a90358cc04ecd
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a33dc6b2ff48444e975067211f049798(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_158a36528405f0ad9565777f4bcaeb43
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_126c6fbcaf38907f6036efba3eadbcdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_158a36528405f0ad9565777f4bcaeb43
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fcbcba2935958716710d034ddbe99ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_158a36528405f0ad9565777f4bcaeb43
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_87a598981badcbb7bdf3490f87564dfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_158a36528405f0ad9565777f4bcaeb43
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3c97f7d329c1958ab1750e2c1cabdaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3c97f7d329c1958ab1750e2c1cabdaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4b77288f5f4247b0b73b95efaaf1f811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4b77288f5f4247b0b73b95efaaf1f811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_808ff10ef9f3ec30a5f924d93cdccb5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_808ff10ef9f3ec30a5f924d93cdccb5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_197d9a6b4a39f94e8be99415bfeb8b4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_197d9a6b4a39f94e8be99415bfeb8b4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b54ff68a7c9eedd51417888a9f83b8c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e890e99ed08a485529a90358cc04ecd
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_929060b89bb6e3d2ac2f93b9f5d55fca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_158a36528405f0ad9565777f4bcaeb43
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_64a1e6dd88ca378f4513e5d32c990dc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e890e99ed08a485529a90358cc04ecd
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f472b4ff88ab4794f5dbb16da25e9754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_158a36528405f0ad9565777f4bcaeb43
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_0be844283dbabd52aca887401b38625a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.pow(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_81d411aff51b77a17c0b41ac2fa281a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be844283dbabd52aca887401b38625a
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cf918fcf87f25e7a251fdc6ddf171026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be844283dbabd52aca887401b38625a
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32f2d53585c7ad97ea7caf93c2964a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af1b687286294a12e674b666c4d58685
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1a3af0299683e8608063185208b89f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8f5130c60f5f754b2570eb2cfae8317
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.47760626673698425], [0.44666361808776855], [0.4991475045681], [0.3156026005744934], [0.29203882813453674], [0.4573269784450531]]], dtype='float32').reshape([1, 6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32f2d53585c7ad97ea7caf93c2964a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af1b687286294a12e674b666c4d58685
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b4e663e188dee53c2361cc945826fb77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8f5130c60f5f754b2570eb2cfae8317
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.15016944706439972], [0.14966975152492523], [0.0473039448261261], [0.27084267139434814], [0.40264034271240234], [0.18267758190631866]]], dtype='float32').reshape([1, 6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_94f6cca6752062f320d3f64bdb49b5ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be844283dbabd52aca887401b38625a
    def get_inputs(self):
        return [
            paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6155bd07f0d92007e131b7b88975cbc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be844283dbabd52aca887401b38625a
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba7941cbc9230dde5ef4efeb2b3f21e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af1b687286294a12e674b666c4d58685
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4e79655f9dc87918025c703a84fe24c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be844283dbabd52aca887401b38625a
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_95fb448657cfeb9007b3bcae710b2837(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be844283dbabd52aca887401b38625a
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3c97f7d329c1958ab1750e2c1cabdaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3c97f7d329c1958ab1750e2c1cabdaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4b77288f5f4247b0b73b95efaaf1f811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4b77288f5f4247b0b73b95efaaf1f811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_808ff10ef9f3ec30a5f924d93cdccb5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_808ff10ef9f3ec30a5f924d93cdccb5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_197d9a6b4a39f94e8be99415bfeb8b4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_197d9a6b4a39f94e8be99415bfeb8b4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f9f56610d9020904a05d2580f1f3ffdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f9f56610d9020904a05d2580f1f3ffdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e1baca7b536edd20c6ff2034426ea9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e1baca7b536edd20c6ff2034426ea9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_08cb44b7f3bd70ccb166f0d7776cb43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_08cb44b7f3bd70ccb166f0d7776cb43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c44eafc82106fb55a87674a701bc400f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c44eafc82106fb55a87674a701bc400f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f9f56610d9020904a05d2580f1f3ffdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f9f56610d9020904a05d2580f1f3ffdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e1baca7b536edd20c6ff2034426ea9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e1baca7b536edd20c6ff2034426ea9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_08cb44b7f3bd70ccb166f0d7776cb43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_08cb44b7f3bd70ccb166f0d7776cb43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c44eafc82106fb55a87674a701bc400f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c44eafc82106fb55a87674a701bc400f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_677ac373f3dc564b908706c622bf5c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be844283dbabd52aca887401b38625a
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_688c8d67ee3eb02655158be0e5524c14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be844283dbabd52aca887401b38625a
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1171ff03ef14d21ff4499042e07e8f4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be844283dbabd52aca887401b38625a
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8375a6f5054c2e4e3d566ffe7ea8ab45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be844283dbabd52aca887401b38625a
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f436a567afa6532cd31fde925a7353cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be844283dbabd52aca887401b38625a
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61320214ae2a42f5b7f14f1b0cce9249(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be844283dbabd52aca887401b38625a
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3c97f7d329c1958ab1750e2c1cabdaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3c97f7d329c1958ab1750e2c1cabdaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4b77288f5f4247b0b73b95efaaf1f811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4b77288f5f4247b0b73b95efaaf1f811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_808ff10ef9f3ec30a5f924d93cdccb5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_808ff10ef9f3ec30a5f924d93cdccb5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_197d9a6b4a39f94e8be99415bfeb8b4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_197d9a6b4a39f94e8be99415bfeb8b4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eca95b82ee9cd5baddbab489e1a0bd17
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d1e0641956040d5b70ae96405ee4d417(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be844283dbabd52aca887401b38625a
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_81d411aff51b77a17c0b41ac2fa281a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be844283dbabd52aca887401b38625a
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b17c993093611236978237bd829a00ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be844283dbabd52aca887401b38625a
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a23bd38f527bb0510f68c7939afe4aa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0be844283dbabd52aca887401b38625a
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()