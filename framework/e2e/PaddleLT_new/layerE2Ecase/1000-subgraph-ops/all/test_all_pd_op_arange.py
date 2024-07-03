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
class PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2(InstanceTrait, paddle.nn.Layer):
    
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


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b7da14c22824be33f66e387344a5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b7da14c22824be33f66e387344a5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d548e45d64736b28fa2c33d5163cafc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d548e45d64736b28fa2c33d5163cafc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c99e6319ece716652da1c17258b1a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c99e6319ece716652da1c17258b1a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]



class PrimitiveOp_d00b43386e8064d2da256f0e484275c5(InstanceTrait, paddle.nn.Layer):
    
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


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d51cd95b0c51b330ee8e7c5fc774c98f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([20.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7f2e102b6ce1c8d734b2618e20c10a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([96.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_affd804ba6f3bfa97c59136134c671f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([48.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd7d71d4ae48179803255ea2fa9e1dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([24.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_baa4fc4bc555a0fe96a136c78e8948d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ac0fc9ff1d429487ca251b313f0d5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ef92703327d66f99367c5fcd61cf6d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_580f9c2bdd6af52cb9d8a7089687622c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([80.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_51b472aaae6e9c5d74d28e46e06a5f83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([40.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d51cd95b0c51b330ee8e7c5fc774c98f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([20.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b7da14c22824be33f66e387344a5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b7da14c22824be33f66e387344a5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d548e45d64736b28fa2c33d5163cafc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d548e45d64736b28fa2c33d5163cafc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c99e6319ece716652da1c17258b1a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c99e6319ece716652da1c17258b1a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_580f9c2bdd6af52cb9d8a7089687622c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([80.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed6bb23219f11957ad47d7a6be397fe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([14.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80e8bbe88ab18b9ace6555266685c499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([28.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc5a35a3664e419c8c9bf4a58f03c6c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([56.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d51cd95b0c51b330ee8e7c5fc774c98f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([20.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_895476acd56aad5694ad0236d0944628(InstanceTrait, paddle.nn.Layer):
    
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


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63307264928768435bdb4b1de8f81ed6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_895476acd56aad5694ad0236d0944628
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([24.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4a6cf34cbe478ff86ff8f9bc5932a34d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([68.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b1480cd640a46b2f7da343cce45c399(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([34.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96221360c2197bfea5b379112979db45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([17.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c99e6319ece716652da1c17258b1a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c99e6319ece716652da1c17258b1a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d548e45d64736b28fa2c33d5163cafc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d548e45d64736b28fa2c33d5163cafc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b7da14c22824be33f66e387344a5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b7da14c22824be33f66e387344a5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_548f6beeefe5765cd8949df383cafe0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_548f6beeefe5765cd8949df383cafe0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1d859d8ea8aaacd55aa622ceb8f5e083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1d859d8ea8aaacd55aa622ceb8f5e083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e45e4e90b368a2b031828fa9a93c01aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([152.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dd46702df1eed9c89fcf959e8350d741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([100.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f64f53ab4d5943a304fe6099a22d2cfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([76.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5ad1d1d5edfc00a4e605786304827bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([50.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_91538a6611cb70c03995e06f65526ffd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([38.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4e3c63367eaf7fc93f0e5fd791d8b9e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([25.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dda3a2b2c65502217b29ab53d8128794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([19.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc262f8b09bb3a78013426a0f416b87b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([13.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fcef38c3220365c979a719ebac6a9cab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_56ca7602989b81f317fbd02dcd4d970e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b7da14c22824be33f66e387344a5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b7da14c22824be33f66e387344a5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d548e45d64736b28fa2c33d5163cafc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d548e45d64736b28fa2c33d5163cafc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c99e6319ece716652da1c17258b1a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c99e6319ece716652da1c17258b1a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e65ae64b5fe651c69b67612a14c324d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([72.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_199586eb6e6ccef2aec7c356d48924a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([36.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c440919eec37e84d8c92d7305c2cd9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00b43386e8064d2da256f0e484275c5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([18.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b7da14c22824be33f66e387344a5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b7da14c22824be33f66e387344a5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d548e45d64736b28fa2c33d5163cafc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d548e45d64736b28fa2c33d5163cafc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c99e6319ece716652da1c17258b1a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c99e6319ece716652da1c17258b1a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e29d1c65e0ab5c52f7d0e99a34bcc2
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]



class PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f(InstanceTrait, paddle.nn.Layer):
    
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


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38fe38e1246339d0f583be2ee6d6eaf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38fe38e1246339d0f583be2ee6d6eaf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31dd29821a671b6a269458615fc51a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31dd29821a671b6a269458615fc51a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89c3b2665c9f067bd7b9140059ab5d51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89c3b2665c9f067bd7b9140059ab5d51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]



class PrimitiveOp_39604a944b3af1244779faef179d12cb(InstanceTrait, paddle.nn.Layer):
    
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


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8b162deb6de6120f1495474670969c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([20.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_42e12d519e764822c5c9936846f2a432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([96.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efc5bdad4a2dd4eac45edcd6444f3f1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([48.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_50bd90f6349b7b5f00c62dd0b01ce09c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([24.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cf416bc1599d3e96f766abf97042e78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eb8c54889c1fa73adfcbda25c038b9ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a299adee7d147342d78a457c57a339ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1a25cf6c5d38da30c85b58883884797d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([80.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1c61ccaef50ddb0042ac9a5a2bad171e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([40.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8b162deb6de6120f1495474670969c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([20.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38fe38e1246339d0f583be2ee6d6eaf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38fe38e1246339d0f583be2ee6d6eaf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31dd29821a671b6a269458615fc51a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31dd29821a671b6a269458615fc51a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89c3b2665c9f067bd7b9140059ab5d51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89c3b2665c9f067bd7b9140059ab5d51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1a25cf6c5d38da30c85b58883884797d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([80.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8896001996e7d67297b9d481797c6f9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([14.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d28f612c48d184167cf2776376b655d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([28.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bfa1aeab8a7d2c4d682bb0e1e2ac6646(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([56.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8b162deb6de6120f1495474670969c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([20.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_88da7208e16fd8beb7f3962f8a2291f0(InstanceTrait, paddle.nn.Layer):
    
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


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_95eb68082d7f82c1eb251ee5ee186d86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88da7208e16fd8beb7f3962f8a2291f0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([24.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99ae502f499901a71f2299b30fbc99f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([68.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e26289f8e5638bc0ba3969dde5c0d0d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([34.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6090ff84ecbd43f694c1d64aa067b65c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([17.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89c3b2665c9f067bd7b9140059ab5d51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89c3b2665c9f067bd7b9140059ab5d51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31dd29821a671b6a269458615fc51a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31dd29821a671b6a269458615fc51a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38fe38e1246339d0f583be2ee6d6eaf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38fe38e1246339d0f583be2ee6d6eaf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5f783d9f3e55d0b8b79619f15204fc5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5f783d9f3e55d0b8b79619f15204fc5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(16, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_65a50acc6e9feee55df95bb89802cf2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_65a50acc6e9feee55df95bb89802cf2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(8, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e562c662ccf162eb90492391574f175f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([152.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dcbbd1a35845e0a0a1502f67188f27b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([100.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b105b989b0b5ac1b11471379b755bb24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([76.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d49758b4051e29418e849a4e0e850485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([50.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d37f1063fa690861161d3141d238f9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([38.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c79e328188687dc004091a8ce6961e24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([25.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_714f809ed4c323d2849dd6ce54c146d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([19.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe53ac9d74cb122cdd63732c5d6bab8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([13.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa3cd7e19081c6540dc3b58abf0c1ac7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e23f419a71284f171e3489d03a83692a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38fe38e1246339d0f583be2ee6d6eaf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38fe38e1246339d0f583be2ee6d6eaf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31dd29821a671b6a269458615fc51a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31dd29821a671b6a269458615fc51a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89c3b2665c9f067bd7b9140059ab5d51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89c3b2665c9f067bd7b9140059ab5d51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0293b59876d8491fff9e9ee7b86697c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([72.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5e5892d768ddc1e56192e92174208390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([36.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_813826396d23cac986769a9731ffb293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39604a944b3af1244779faef179d12cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([18.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38fe38e1246339d0f583be2ee6d6eaf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38fe38e1246339d0f583be2ee6d6eaf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(32, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31dd29821a671b6a269458615fc51a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b31dd29821a671b6a269458615fc51a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(64, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89c3b2665c9f067bd7b9140059ab5d51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89c3b2665c9f067bd7b9140059ab5d51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b93d40e90b1f4f5aea5290532abfd32f
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor(128, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()