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
class PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_45590ccb067a993036bdf78e23f474d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_45590ccb067a993036bdf78e23f474d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c87c6f0497ecd74dc60320a3e3989617(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c87c6f0497ecd74dc60320a3e3989617(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7380487db5821129bd5686cd6168f44f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7380487db5821129bd5686cd6168f44f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78e91db0670cae193d8e61f90e54e9f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78e91db0670cae193d8e61f90e54e9f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4556dc993ddd4a2f6294eb744c00b5d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4556dc993ddd4a2f6294eb744c00b5d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed00b2d0aca2fd7a1a6984cd38dd93a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed00b2d0aca2fd7a1a6984cd38dd93a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_45590ccb067a993036bdf78e23f474d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_45590ccb067a993036bdf78e23f474d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7cec3d092d6daee0483bfbca6289a2a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7cec3d092d6daee0483bfbca6289a2a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_16c9b5aa2d1049a86ea83c39e582d173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_16c9b5aa2d1049a86ea83c39e582d173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4babea563f0cfc94cbd543704a0cd693(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2ef7b46ffa2e7802e1eebc4577f565a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2ef7b46ffa2e7802e1eebc4577f565a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2ef7b46ffa2e7802e1eebc4577f565a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2ef7b46ffa2e7802e1eebc4577f565a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96bdc15605e030639836217908d584b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96bdc15605e030639836217908d584b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_73076222e0df50639e7596a6fc86f7a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_73076222e0df50639e7596a6fc86f7a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e4e1cc73ae905136448e537df1b64ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e4e1cc73ae905136448e537df1b64ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_135bb2deaaeb09d53a1b74581f85f946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_135bb2deaaeb09d53a1b74581f85f946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_dbb39dd6deed0358d29234ca99e486af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c37ecaf400ad4bcdbb0320c838f09673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39208778738975525], [0.2094176560640335], [0.1283661276102066], [0.27551141381263733], [0.03479340672492981], [0.3570266664028168], [0.11812948435544968], [0.2456916868686676], [0.48227444291114807]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.1547079086303711], [0.3591558635234833], [0.15294159948825836], [0.019754718989133835], [0.2135521024465561], [0.2554653286933899], [0.3142800033092499], [0.4077095687389374], [0.42899003624916077]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_555190aab4f35b8fc320ed6848247ec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.49722179770469666], [0.4310191571712494], [0.3834795355796814], [0.30722612142562866], [0.2178652435541153], [0.11710164695978165], [0.23001685738563538], [0.22062554955482483], [0.498458594083786]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.39656439423561096], [0.12202204018831253], [0.0054341829381883144], [0.014495838433504105], [0.0013494952581822872], [0.28443190455436707], [0.2855451703071594], [0.20876038074493408], [0.38022154569625854]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6bd5f04137aaa1189b3d7b33e76dae34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36820653080940247], [0.03848660737276077], [0.08805205672979355], [0.03780742734670639], [0.23160281777381897], [0.18210230767726898], [0.21688763797283173], [0.009429873898625374], [0.23962129652500153]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.08568544685840607], [0.1383717805147171], [0.2306109517812729], [0.03215041384100914], [0.05588645488023758], [0.1058163195848465], [0.12910811603069305], [0.02052994631230831], [0.24312002956867218]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13987dd235c7dd4ccf91ea333aae1aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39371946454048157], [0.050644610077142715], [0.09682576358318329], [0.37938541173934937], [0.37218934297561646], [0.31943804025650024], [0.009334404021501541], [0.027887742966413498], [0.1417359560728073]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.08652636408805847], [0.38375985622406006], [0.22444428503513336], [0.3983890116214752], [0.28843000531196594], [0.07827463001012802], [0.3716101050376892], [0.20676524937152863], [0.260421484708786]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_51e7e195b86c93bf340055a938ec9edf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_51e7e195b86c93bf340055a938ec9edf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_51e7e195b86c93bf340055a938ec9edf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_51e7e195b86c93bf340055a938ec9edf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_b15114e4826950e42bafd62b41493dba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_edfa362723727057f9d5baf7fcda533a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b15114e4826950e42bafd62b41493dba
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1027945876121521, 0.3881237208843231, 0.152043879032135, 0.21335071325302124, 0.38008421659469604, 0.12074317038059235], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2608264982700348, 0.4725295603275299, 0.30785173177719116, 0.08038189262151718, 0.0575774721801281, 0.3921468257904053], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cacd5352e870d3d2b035786152cccb2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b15114e4826950e42bafd62b41493dba
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2527327835559845, 0.1542942374944687, 0.470980703830719, 0.32545068860054016, 0.27355799078941345, 0.35376250743865967], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4957810342311859, 0.08107513189315796, 0.33436471223831177, 0.4922436773777008, 0.36378422379493713, 0.13394984602928162], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a9764080b3e24ddfa6616ff2927e78d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b15114e4826950e42bafd62b41493dba
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1027945876121521, 0.3881237208843231, 0.152043879032135, 0.21335071325302124, 0.38008421659469604, 0.12074317038059235], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2304920107126236, 0.0650036409497261, 0.457919716835022, 0.030212104320526123, 0.35011738538742065, 0.12877273559570312], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_940105d6e5e0c0aaae3c44f00cca0b05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b15114e4826950e42bafd62b41493dba
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2527327835559845, 0.1542942374944687, 0.470980703830719, 0.32545068860054016, 0.27355799078941345, 0.35376250743865967], dtype='float32').reshape([6]),
            paddle.to_tensor([0.23068910837173462, 0.1622818112373352, 0.3164260983467102, 0.16976673901081085, 0.15035927295684814, 0.22009573876857758], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f27ff3b8f310e2af4f7636a3ba7247f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b15114e4826950e42bafd62b41493dba
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2608264982700348, 0.4725295603275299, 0.30785173177719116, 0.21335071325302124, 0.38008421659469604, 0.3921468257904053], dtype='float32').reshape([6]),
            paddle.to_tensor([0.24978351593017578, 0.11446373909711838, 0.06451474130153656, 0.04623079672455788, 0.3613818883895874, 0.4987070560455322], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f31cd8ac229ff2c6d374b493246969dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b15114e4826950e42bafd62b41493dba
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4957810342311859, 0.1542942374944687, 0.470980703830719, 0.4922436773777008, 0.36378422379493713, 0.35376250743865967], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2845926582813263, 0.16627027094364166, 0.25496557354927063, 0.29948753118515015, 0.21094956994056702, 0.2771330177783966], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d2a7a47c32bd2b6f0e20291b3f3db708(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d2a7a47c32bd2b6f0e20291b3f3db708(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d2a7a47c32bd2b6f0e20291b3f3db708(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d2a7a47c32bd2b6f0e20291b3f3db708(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_7da8fae49389fd8247a4e00f8c89a5aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_79dc8921e550a1c5e521294d2d55d922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7da8fae49389fd8247a4e00f8c89a5aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.000000013351432e-10], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78e91db0670cae193d8e61f90e54e9f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78e91db0670cae193d8e61f90e54e9f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_73076222e0df50639e7596a6fc86f7a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_73076222e0df50639e7596a6fc86f7a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa0eff6667c70c3ca420506348d601a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa0eff6667c70c3ca420506348d601a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa0eff6667c70c3ca420506348d601a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa0eff6667c70c3ca420506348d601a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed00b2d0aca2fd7a1a6984cd38dd93a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed00b2d0aca2fd7a1a6984cd38dd93a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c87c6f0497ecd74dc60320a3e3989617(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c87c6f0497ecd74dc60320a3e3989617(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8cf3cb0aa1bf00632421a49b4f1a500b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2977246940135956]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3555246889591217]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_059fa337c06428aa0fcd0398a0d1e07a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.45340871810913086]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.43772584199905396]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7066b8c765ae2ddba3bf17f65de7cd62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33794644474983215]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.11903903633356094]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e96314e76c0c2c944c8f476fc2854385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12983585894107819]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0831770971417427]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c536977ff50e52181ebebcb3ff273b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15491808950901031], [0.16919776797294617], [0.1292070597410202], [0.12714418768882751], [0.33482757210731506], [0.2098187357187271]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3895230293273926], [0.4666270613670349], [0.17671829462051392], [0.3581591248512268], [0.42263373732566833], [0.24344341456890106]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3454176afd0defd4410eb4bc8dbbc8d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4103717803955078], [0.39778658747673035], [0.40150901675224304], [0.42183321714401245], [0.09941722452640533], [0.25498807430267334]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.1010279506444931], [0.2909882366657257], [0.2298392504453659], [0.29276591539382935], [0.1773831695318222], [0.48681026697158813]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_90d8fcfe6621ea1813d43fd71e4b50a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20998290181159973], [0.4498758018016815], [0.24683597683906555], [0.20288529992103577], [0.21599259972572327], [0.19651634991168976]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.22682112455368042], [0.18442489206790924], [0.1968512237071991], [0.46012169122695923], [0.40794169902801514], [0.4862612187862396]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_551ccc662117a704cf91111c93396095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4901370406150818], [0.2509276866912842], [0.2429288774728775], [0.06132770702242851], [0.08567338436841965], [0.23274247348308563]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.4151418209075928], [0.3606884777545929], [0.13929075002670288], [0.03818127512931824], [0.28175589442253113], [0.3325986862182617]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96bdc15605e030639836217908d584b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96bdc15605e030639836217908d584b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7380487db5821129bd5686cd6168f44f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7380487db5821129bd5686cd6168f44f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d2667353754bde5a971c7c5c6ee75e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d2667353754bde5a971c7c5c6ee75e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d2667353754bde5a971c7c5c6ee75e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d2667353754bde5a971c7c5c6ee75e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5a3107e3786d3d3041daed12bfcb22d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5a3107e3786d3d3041daed12bfcb22d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2345143a485b35a707265ac765f14573(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2345143a485b35a707265ac765f14573(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2345143a485b35a707265ac765f14573(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2345143a485b35a707265ac765f14573(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ae03cf1c9f27e7c079f956cbedf43373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ae03cf1c9f27e7c079f956cbedf43373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ae03cf1c9f27e7c079f956cbedf43373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ae03cf1c9f27e7c079f956cbedf43373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4556dc993ddd4a2f6294eb744c00b5d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4556dc993ddd4a2f6294eb744c00b5d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e4e1cc73ae905136448e537df1b64ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e4e1cc73ae905136448e537df1b64ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6e60c7f68e36dffe2a13ad329c308df9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3776218891143799], [0.017230553552508354], [0.12865380942821503], [0.0789160430431366], [0.29648688435554504]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4706946909427643], [0.39319300651550293], [0.19910383224487305], [0.3212777376174927], [0.27562597393989563]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_48a236f43711dd32c221f6c64acded21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04191706329584122], [0.1894516497850418], [0.11870089173316956], [0.0004433089052326977], [0.28320610523223877]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.16876435279846191], [0.1280040442943573], [0.3004133403301239], [0.4173230230808258], [0.3782731890678406]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1505e0021d50db14db121f065a773551(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3077138066291809], [0.21676185727119446], [0.36836960911750793], [0.01315593346953392], [0.1262916475534439]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.27102693915367126], [0.3185052275657654], [0.3153221905231476], [0.430990993976593], [0.4189223051071167]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c4c6be31e2e7640df4338be3b3a25b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39167019724845886], [0.046127624809741974], [0.13554736971855164], [0.2480829954147339], [0.17310704290866852]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4086301028728485], [0.391907662153244], [0.04106692224740982], [0.014858669601380825], [0.4931931793689728]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_16c9b5aa2d1049a86ea83c39e582d173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_16c9b5aa2d1049a86ea83c39e582d173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_135bb2deaaeb09d53a1b74581f85f946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_135bb2deaaeb09d53a1b74581f85f946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_72017358536c023689b8ebeb0b3719a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_72017358536c023689b8ebeb0b3719a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dd031253a2f53a989acd5f6a6c4cdef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dd031253a2f53a989acd5f6a6c4cdef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dd031253a2f53a989acd5f6a6c4cdef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dd031253a2f53a989acd5f6a6c4cdef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e70b18fd08fd66974dbe301f40c3b323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e70b18fd08fd66974dbe301f40c3b323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e70b18fd08fd66974dbe301f40c3b323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e70b18fd08fd66974dbe301f40c3b323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a6686765135b072a0365c126e1188ee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a6686765135b072a0365c126e1188ee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a6686765135b072a0365c126e1188ee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a6686765135b072a0365c126e1188ee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5a3107e3786d3d3041daed12bfcb22d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5a3107e3786d3d3041daed12bfcb22d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_623e127ff9062a2f5f8592a5eae633d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4710065722465515], [0.27305787801742554], [0.1443967968225479], [0.18359437584877014]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.11861217021942139], [0.1668710708618164], [0.37650251388549805], [0.3584076464176178]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_87d4f1e8f1e8f5e8822fdff511f00da2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3558596074581146], [0.24631327390670776], [0.38240188360214233], [0.43209323287010193]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.17836087942123413], [0.34534937143325806], [0.45495909452438354], [0.29443272948265076]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4560330008a1e8c9b2485d5784b1ef39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1317010074853897], [0.1832035928964615], [0.005624867510050535], [0.382903516292572]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.21841885149478912], [0.4341205954551697], [0.17934243381023407], [0.48898959159851074]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_634a22fd2be9ed7ea7f6e6157a08aba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.407511442899704], [0.41122034192085266], [0.36212438344955444], [0.47452500462532043]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2849007844924927], [0.1524905115365982], [0.386353075504303], [0.4171924889087677]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_936c4584cb51a8986799a6d4249800da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_936c4584cb51a8986799a6d4249800da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_936c4584cb51a8986799a6d4249800da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_936c4584cb51a8986799a6d4249800da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_72017358536c023689b8ebeb0b3719a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_72017358536c023689b8ebeb0b3719a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7cec3d092d6daee0483bfbca6289a2a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7cec3d092d6daee0483bfbca6289a2a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1b2edee8f90f643bd71e6251333740a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1b2edee8f90f643bd71e6251333740a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5953813d6bf2b7055fc3be5d274ff1e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5953813d6bf2b7055fc3be5d274ff1e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5953813d6bf2b7055fc3be5d274ff1e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5953813d6bf2b7055fc3be5d274ff1e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4babea563f0cfc94cbd543704a0cd693
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1b2edee8f90f643bd71e6251333740a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1b2edee8f90f643bd71e6251333740a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_45590ccb067a993036bdf78e23f474d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_45590ccb067a993036bdf78e23f474d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c87c6f0497ecd74dc60320a3e3989617(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c87c6f0497ecd74dc60320a3e3989617(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7380487db5821129bd5686cd6168f44f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7380487db5821129bd5686cd6168f44f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78e91db0670cae193d8e61f90e54e9f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78e91db0670cae193d8e61f90e54e9f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4556dc993ddd4a2f6294eb744c00b5d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4556dc993ddd4a2f6294eb744c00b5d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed00b2d0aca2fd7a1a6984cd38dd93a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed00b2d0aca2fd7a1a6984cd38dd93a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_45590ccb067a993036bdf78e23f474d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_45590ccb067a993036bdf78e23f474d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7cec3d092d6daee0483bfbca6289a2a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7cec3d092d6daee0483bfbca6289a2a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_16c9b5aa2d1049a86ea83c39e582d173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_16c9b5aa2d1049a86ea83c39e582d173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8ba9c0bc1fe9c11ceee2917c4527ba7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8ba9c0bc1fe9c11ceee2917c4527ba7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8ba9c0bc1fe9c11ceee2917c4527ba7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8ba9c0bc1fe9c11ceee2917c4527ba7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96bdc15605e030639836217908d584b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96bdc15605e030639836217908d584b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_73076222e0df50639e7596a6fc86f7a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_73076222e0df50639e7596a6fc86f7a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e4e1cc73ae905136448e537df1b64ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e4e1cc73ae905136448e537df1b64ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_135bb2deaaeb09d53a1b74581f85f946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_135bb2deaaeb09d53a1b74581f85f946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c37ecaf400ad4bcdbb0320c838f09673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39208778738975525], [0.2094176560640335], [0.1283661276102066], [0.27551141381263733], [0.03479340672492981], [0.3570266664028168], [0.11812948435544968], [0.2456916868686676], [0.48227444291114807]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.1547079086303711], [0.3591558635234833], [0.15294159948825836], [0.019754718989133835], [0.2135521024465561], [0.2554653286933899], [0.3142800033092499], [0.4077095687389374], [0.42899003624916077]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_555190aab4f35b8fc320ed6848247ec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.49722179770469666], [0.4310191571712494], [0.3834795355796814], [0.30722612142562866], [0.2178652435541153], [0.11710164695978165], [0.23001685738563538], [0.22062554955482483], [0.498458594083786]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.39656439423561096], [0.12202204018831253], [0.0054341829381883144], [0.014495838433504105], [0.0013494952581822872], [0.28443190455436707], [0.2855451703071594], [0.20876038074493408], [0.38022154569625854]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6bd5f04137aaa1189b3d7b33e76dae34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36820653080940247], [0.03848660737276077], [0.08805205672979355], [0.03780742734670639], [0.23160281777381897], [0.18210230767726898], [0.21688763797283173], [0.009429873898625374], [0.23962129652500153]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.08568544685840607], [0.1383717805147171], [0.2306109517812729], [0.03215041384100914], [0.05588645488023758], [0.1058163195848465], [0.12910811603069305], [0.02052994631230831], [0.24312002956867218]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13987dd235c7dd4ccf91ea333aae1aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39371946454048157], [0.050644610077142715], [0.09682576358318329], [0.37938541173934937], [0.37218934297561646], [0.31943804025650024], [0.009334404021501541], [0.027887742966413498], [0.1417359560728073]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.08652636408805847], [0.38375985622406006], [0.22444428503513336], [0.3983890116214752], [0.28843000531196594], [0.07827463001012802], [0.3716101050376892], [0.20676524937152863], [0.260421484708786]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5d1fc87e695d15078c9c7b6003082b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5d1fc87e695d15078c9c7b6003082b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5d1fc87e695d15078c9c7b6003082b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5d1fc87e695d15078c9c7b6003082b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_edfa362723727057f9d5baf7fcda533a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b15114e4826950e42bafd62b41493dba
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1027945876121521, 0.3881237208843231, 0.152043879032135, 0.21335071325302124, 0.38008421659469604, 0.12074317038059235], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2608264982700348, 0.4725295603275299, 0.30785173177719116, 0.08038189262151718, 0.0575774721801281, 0.3921468257904053], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cacd5352e870d3d2b035786152cccb2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b15114e4826950e42bafd62b41493dba
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2527327835559845, 0.1542942374944687, 0.470980703830719, 0.32545068860054016, 0.27355799078941345, 0.35376250743865967], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4957810342311859, 0.08107513189315796, 0.33436471223831177, 0.4922436773777008, 0.36378422379493713, 0.13394984602928162], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a9764080b3e24ddfa6616ff2927e78d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b15114e4826950e42bafd62b41493dba
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1027945876121521, 0.3881237208843231, 0.152043879032135, 0.21335071325302124, 0.38008421659469604, 0.12074317038059235], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2304920107126236, 0.0650036409497261, 0.457919716835022, 0.030212104320526123, 0.35011738538742065, 0.12877273559570312], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_940105d6e5e0c0aaae3c44f00cca0b05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b15114e4826950e42bafd62b41493dba
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2527327835559845, 0.1542942374944687, 0.470980703830719, 0.32545068860054016, 0.27355799078941345, 0.35376250743865967], dtype='float32').reshape([6]),
            paddle.to_tensor([0.23068910837173462, 0.1622818112373352, 0.3164260983467102, 0.16976673901081085, 0.15035927295684814, 0.22009573876857758], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f27ff3b8f310e2af4f7636a3ba7247f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b15114e4826950e42bafd62b41493dba
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2608264982700348, 0.4725295603275299, 0.30785173177719116, 0.21335071325302124, 0.38008421659469604, 0.3921468257904053], dtype='float32').reshape([6]),
            paddle.to_tensor([0.24978351593017578, 0.11446373909711838, 0.06451474130153656, 0.04623079672455788, 0.3613818883895874, 0.4987070560455322], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f31cd8ac229ff2c6d374b493246969dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b15114e4826950e42bafd62b41493dba
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4957810342311859, 0.1542942374944687, 0.470980703830719, 0.4922436773777008, 0.36378422379493713, 0.35376250743865967], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2845926582813263, 0.16627027094364166, 0.25496557354927063, 0.29948753118515015, 0.21094956994056702, 0.2771330177783966], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f13fc198cac4668a0001829d063d3a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f13fc198cac4668a0001829d063d3a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f13fc198cac4668a0001829d063d3a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f13fc198cac4668a0001829d063d3a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_298b3063e6477aaa93ddc21174d33b57(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ef20dd23028bc1e154fa6f6a15e14f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_298b3063e6477aaa93ddc21174d33b57
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.000000013351432e-10], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78e91db0670cae193d8e61f90e54e9f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78e91db0670cae193d8e61f90e54e9f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_73076222e0df50639e7596a6fc86f7a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_73076222e0df50639e7596a6fc86f7a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d18928ea57b76008ab30485bf141f15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d18928ea57b76008ab30485bf141f15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d18928ea57b76008ab30485bf141f15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d18928ea57b76008ab30485bf141f15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed00b2d0aca2fd7a1a6984cd38dd93a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed00b2d0aca2fd7a1a6984cd38dd93a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c87c6f0497ecd74dc60320a3e3989617(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c87c6f0497ecd74dc60320a3e3989617(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8cf3cb0aa1bf00632421a49b4f1a500b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2977246940135956]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3555246889591217]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_059fa337c06428aa0fcd0398a0d1e07a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.45340871810913086]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.43772584199905396]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7066b8c765ae2ddba3bf17f65de7cd62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33794644474983215]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.11903903633356094]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e96314e76c0c2c944c8f476fc2854385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12983585894107819]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0831770971417427]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c536977ff50e52181ebebcb3ff273b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15491808950901031], [0.16919776797294617], [0.1292070597410202], [0.12714418768882751], [0.33482757210731506], [0.2098187357187271]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3895230293273926], [0.4666270613670349], [0.17671829462051392], [0.3581591248512268], [0.42263373732566833], [0.24344341456890106]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3454176afd0defd4410eb4bc8dbbc8d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4103717803955078], [0.39778658747673035], [0.40150901675224304], [0.42183321714401245], [0.09941722452640533], [0.25498807430267334]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.1010279506444931], [0.2909882366657257], [0.2298392504453659], [0.29276591539382935], [0.1773831695318222], [0.48681026697158813]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_90d8fcfe6621ea1813d43fd71e4b50a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20998290181159973], [0.4498758018016815], [0.24683597683906555], [0.20288529992103577], [0.21599259972572327], [0.19651634991168976]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.22682112455368042], [0.18442489206790924], [0.1968512237071991], [0.46012169122695923], [0.40794169902801514], [0.4862612187862396]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_551ccc662117a704cf91111c93396095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4901370406150818], [0.2509276866912842], [0.2429288774728775], [0.06132770702242851], [0.08567338436841965], [0.23274247348308563]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.4151418209075928], [0.3606884777545929], [0.13929075002670288], [0.03818127512931824], [0.28175589442253113], [0.3325986862182617]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96bdc15605e030639836217908d584b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96bdc15605e030639836217908d584b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7380487db5821129bd5686cd6168f44f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7380487db5821129bd5686cd6168f44f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b18cfb9dc8f2931c1f9ec2af08a3eff2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b18cfb9dc8f2931c1f9ec2af08a3eff2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b18cfb9dc8f2931c1f9ec2af08a3eff2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b18cfb9dc8f2931c1f9ec2af08a3eff2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5a3107e3786d3d3041daed12bfcb22d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5a3107e3786d3d3041daed12bfcb22d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61b580e6493f4961c022d53a1c5f0b2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61b580e6493f4961c022d53a1c5f0b2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61b580e6493f4961c022d53a1c5f0b2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61b580e6493f4961c022d53a1c5f0b2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1b75d202644ebc6efec7b2fa5615717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1b75d202644ebc6efec7b2fa5615717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1b75d202644ebc6efec7b2fa5615717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1b75d202644ebc6efec7b2fa5615717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4556dc993ddd4a2f6294eb744c00b5d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4556dc993ddd4a2f6294eb744c00b5d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e4e1cc73ae905136448e537df1b64ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e4e1cc73ae905136448e537df1b64ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6e60c7f68e36dffe2a13ad329c308df9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3776218891143799], [0.017230553552508354], [0.12865380942821503], [0.0789160430431366], [0.29648688435554504]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4706946909427643], [0.39319300651550293], [0.19910383224487305], [0.3212777376174927], [0.27562597393989563]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_48a236f43711dd32c221f6c64acded21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04191706329584122], [0.1894516497850418], [0.11870089173316956], [0.0004433089052326977], [0.28320610523223877]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.16876435279846191], [0.1280040442943573], [0.3004133403301239], [0.4173230230808258], [0.3782731890678406]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1505e0021d50db14db121f065a773551(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3077138066291809], [0.21676185727119446], [0.36836960911750793], [0.01315593346953392], [0.1262916475534439]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.27102693915367126], [0.3185052275657654], [0.3153221905231476], [0.430990993976593], [0.4189223051071167]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c4c6be31e2e7640df4338be3b3a25b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39167019724845886], [0.046127624809741974], [0.13554736971855164], [0.2480829954147339], [0.17310704290866852]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4086301028728485], [0.391907662153244], [0.04106692224740982], [0.014858669601380825], [0.4931931793689728]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_16c9b5aa2d1049a86ea83c39e582d173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_16c9b5aa2d1049a86ea83c39e582d173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_135bb2deaaeb09d53a1b74581f85f946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_135bb2deaaeb09d53a1b74581f85f946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_72017358536c023689b8ebeb0b3719a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_72017358536c023689b8ebeb0b3719a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8f29ce19c8be6a66e81daa18122a9aa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8f29ce19c8be6a66e81daa18122a9aa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8f29ce19c8be6a66e81daa18122a9aa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8f29ce19c8be6a66e81daa18122a9aa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9664bc6a24afc226825121d6e9ad162c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9664bc6a24afc226825121d6e9ad162c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9664bc6a24afc226825121d6e9ad162c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9664bc6a24afc226825121d6e9ad162c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4e296967aaa31c2189a5ee43babe3455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4e296967aaa31c2189a5ee43babe3455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4e296967aaa31c2189a5ee43babe3455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4e296967aaa31c2189a5ee43babe3455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5a3107e3786d3d3041daed12bfcb22d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5a3107e3786d3d3041daed12bfcb22d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_623e127ff9062a2f5f8592a5eae633d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4710065722465515], [0.27305787801742554], [0.1443967968225479], [0.18359437584877014]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.11861217021942139], [0.1668710708618164], [0.37650251388549805], [0.3584076464176178]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_87d4f1e8f1e8f5e8822fdff511f00da2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3558596074581146], [0.24631327390670776], [0.38240188360214233], [0.43209323287010193]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.17836087942123413], [0.34534937143325806], [0.45495909452438354], [0.29443272948265076]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4560330008a1e8c9b2485d5784b1ef39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1317010074853897], [0.1832035928964615], [0.005624867510050535], [0.382903516292572]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.21841885149478912], [0.4341205954551697], [0.17934243381023407], [0.48898959159851074]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_634a22fd2be9ed7ea7f6e6157a08aba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.407511442899704], [0.41122034192085266], [0.36212438344955444], [0.47452500462532043]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2849007844924927], [0.1524905115365982], [0.386353075504303], [0.4171924889087677]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61429a9bcde67331774da4fabf7fa325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61429a9bcde67331774da4fabf7fa325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61429a9bcde67331774da4fabf7fa325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61429a9bcde67331774da4fabf7fa325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_72017358536c023689b8ebeb0b3719a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_72017358536c023689b8ebeb0b3719a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7cec3d092d6daee0483bfbca6289a2a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7cec3d092d6daee0483bfbca6289a2a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1b2edee8f90f643bd71e6251333740a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1b2edee8f90f643bd71e6251333740a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7de8252518296789b105b7a04f3c3ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7de8252518296789b105b7a04f3c3ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7de8252518296789b105b7a04f3c3ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7de8252518296789b105b7a04f3c3ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbb39dd6deed0358d29234ca99e486af
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1b2edee8f90f643bd71e6251333740a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1b2edee8f90f643bd71e6251333740a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c995f632c41c78ba5a6ecb11af0406f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()