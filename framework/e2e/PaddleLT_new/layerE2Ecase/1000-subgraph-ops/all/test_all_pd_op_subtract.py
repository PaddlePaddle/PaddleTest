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
class PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_91a85761701982e53d3c9aefee45ccec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.04714285582304001]], [[0.24956832826137543]], [[0.06141399219632149]], [[0.27293410897254944]], [[0.020540714263916016]], [[0.47400450706481934]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6053178310394287]], [[0.6067324876785278]], [[0.7563060522079468]], [[0.6030699014663696]], [[0.6912596821784973]], [[0.6188269853591919]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_02ce98b787078fb5db949c316ce3dc84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.43553808331489563]], [[0.202225923538208]], [[0.1559573858976364]], [[0.39463528990745544]], [[0.45692548155784607]], [[0.21347668766975403]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6967204213142395]], [[0.5455881357192993]], [[0.774244487285614]], [[0.5381509065628052]], [[0.5265048742294312]], [[0.7456739544868469]]], dtype='float32').reshape([6, 1, 1]),
        ]



class PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a929203e3767fdbf9f28631d4035808b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7837eb9c20d5f70502fabaaf04eced81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_0dd79751e325816d3c3f5495758327f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12096, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_968fbf05dda988a8cd65895aa4dbfa47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dd79751e325816d3c3f5495758327f0
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_af7abc8f3c07a92290a9b6159b7d361b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dabba7fe0741275dfcdc1ed73b9802f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.009778277017176151, 0.02429402805864811]], [[0.0029977455269545317, 0.20873524248600006]], [[0.46495717763900757, 0.0797295793890953]], [[0.39304178953170776, 0.019942941144108772]], [[0.1783885955810547, 0.41458258032798767]], [[0.47772762179374695, 0.20179226994514465]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.358097106218338, 0.35106879472732544]], [[0.4313538074493408, 0.08216709643602371]], [[0.05587208271026611, 0.3657324016094208]], [[0.31227242946624756, 0.3250352442264557]], [[0.4644756019115448, 0.35592329502105713]], [[0.12418679893016815, 0.4918902516365051]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7eb8e2e75859be7d760cee2a82bf9707(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.3234972059726715, 0.20493970811367035]], [[0.3132672607898712, 0.17412687838077545]], [[0.014389144256711006, 0.3884662687778473]], [[0.3867090344429016, 0.06462222337722778]], [[0.17619918286800385, 0.07482553273439407]], [[0.027263877913355827, 0.33704501390457153]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.358097106218338, 0.35106879472732544]], [[0.4313538074493408, 0.08216709643602371]], [[0.05587208271026611, 0.3657324016094208]], [[0.31227242946624756, 0.3250352442264557]], [[0.4644756019115448, 0.35592329502105713]], [[0.12418679893016815, 0.4918902516365051]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]



class PrimitiveOp_c5c8a13052fe23a22c84469701570109(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2074ab14bd0a265b1de2b8bb93fdbf31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5c8a13052fe23a22c84469701570109
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.29133909940719604, 0.4853375554084778]], [[0.36761757731437683, 0.4571003019809723]], [[0.46794551610946655, 0.1688530147075653]], [[0.2670198976993561, 0.1982056200504303]], [[0.12026780098676682, 0.48663008213043213]], [[0.2773101329803467, 0.48897823691368103]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_99bddb5fd63ecd93cc59d50704818428(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55e8ed499f2a3bcd8b91a2a8e96cbbb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
            paddle.to_tensor([0.3780723512172699, 0.10681550204753876, 0.21278166770935059, 0.23138809204101562, 0.32906362414360046, 0.22839735448360443, 0.13309429585933685, 0.0206376351416111, 0.05156761035323143, 0.46504005789756775, 0.13148869574069977, 0.449235200881958, 0.08905600011348724, 0.015455592423677444, 0.3863491714000702, 0.4747481942176819], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_25a7ceff1e2612cd942241b9353c5a2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3780723512172699, 0.10681550204753876, 0.21278166770935059, 0.23138809204101562, 0.32906362414360046, 0.22839735448360443, 0.13309429585933685, 0.0206376351416111, 0.05156761035323143, 0.46504005789756775, 0.13148869574069977, 0.449235200881958, 0.08905600011348724, 0.015455592423677444, 0.3863491714000702, 0.4747481942176819], dtype='float32').reshape([16]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_1d41c4f35c9bcc31e08344f76597024f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            paddle.static.InputSpec(shape=[300], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3862a2257d0a7b8b364fc904b57cdcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d41c4f35c9bcc31e08344f76597024f
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3862a2257d0a7b8b364fc904b57cdcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d41c4f35c9bcc31e08344f76597024f
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6e02c6ca3972717e5b5b70e1ebc41e36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ce10a710f7565f96c00a1389a91699d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c8fcf76e1ccec496fe7d7fa2cf77661e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_686422e12c80392d343002ed877ae454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_686422e12c80392d343002ed877ae454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_686422e12c80392d343002ed877ae454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_686422e12c80392d343002ed877ae454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_686422e12c80392d343002ed877ae454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_686422e12c80392d343002ed877ae454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_686422e12c80392d343002ed877ae454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_686422e12c80392d343002ed877ae454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_686422e12c80392d343002ed877ae454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_686422e12c80392d343002ed877ae454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_686422e12c80392d343002ed877ae454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52d6c0904952e02947e994d334c88237(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38dc692e168c69c6d9272c552d7727e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c8fcf76e1ccec496fe7d7fa2cf77661e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9a650a7e9391931f30b34078c4bdac3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0660882219672203, 0.47173580527305603, 0.263930082321167, 0.1906377077102661], [0.08412470668554306, 0.41810542345046997, 0.11181697994470596, 0.21970145404338837], [0.15435844659805298, 0.23323611915111542, 0.2906814217567444, 0.22473807632923126], [0.43175461888313293, 0.4601755738258362, 0.09878799319267273, 0.04755397140979767], [0.2544768452644348, 0.316487193107605, 0.2239605039358139, 0.27877798676490784]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.4106596112251282, 0.14335723221302032, 0.02330786921083927, 0.11569497734308243], [0.42931506037712097, 0.01977924257516861, 0.230007603764534, 0.057321660220623016], [0.2798386216163635, 0.10481530427932739, 0.4301333725452423, 0.38091546297073364], [0.3798501193523407, 0.11627687513828278, 0.3571585416793823, 0.09328742325305939], [0.4864861071109772, 0.2632262408733368, 0.27916863560676575, 0.29254499077796936]], dtype='float32').reshape([5, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_aedece1b298498e3693fa2b7e488785d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 5376, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_39ff3ebe61c4806058d6c962f4ef9475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aedece1b298498e3693fa2b7e488785d
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1d596efcb9d2732b85dc08040cf23b5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2620193064212799, 0.11581896245479584, 0.053561560809612274, 0.08257486671209335], [0.03958377614617348, 0.3950630724430084, 0.20495222508907318, 0.3324486017227173], [0.3044680953025818, 0.4422242343425751, 0.3617860674858093, 0.3761495053768158], [0.03958377614617348, 0.3950630724430084, 0.20495222508907318, 0.3324486017227173], [0.3044680953025818, 0.4422242343425751, 0.3617860674858093, 0.3761495053768158]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.4025401175022125, 0.3901353180408478, 0.36967140436172485, 0.09798751771450043], [0.3619749844074249, 0.07483921945095062, 0.3436369299888611, 0.2519150972366333], [0.28524336218833923, 0.36912211775779724, 0.19368404150009155, 0.3241569995880127], [0.3619749844074249, 0.07483921945095062, 0.3436369299888611, 0.2519150972366333], [0.28524336218833923, 0.36912211775779724, 0.19368404150009155, 0.3241569995880127]], dtype='float32').reshape([5, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dbf0f22c4f3a7340ebbd00c64a90cf2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08568544685840607], [0.03848660737276077], [0.08805205672979355], [0.03215041384100914], [0.05588645488023758], [0.1058163195848465], [0.12910811603069305], [0.009429873898625374], [0.23962129652500153]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.39208778738975525], [0.3591558635234833], [0.15294159948825836], [0.27551141381263733], [0.2135521024465561], [0.3570266664028168], [0.3142800033092499], [0.4077095687389374], [0.48227444291114807]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1dd8422d0d5d1bc60990131be414319e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08652636408805847], [0.050644610077142715], [0.09682576358318329], [0.37938541173934937], [0.28843000531196594], [0.07827463001012802], [0.009334404021501541], [0.027887742966413498], [0.1417359560728073]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.49722179770469666], [0.4310191571712494], [0.3834795355796814], [0.30722612142562866], [0.2178652435541153], [0.28443190455436707], [0.2855451703071594], [0.22062554955482483], [0.498458594083786]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f3532d8815f2a1e2c847f8d2ee8e106(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36820653080940247], [0.03848660737276077], [0.08805205672979355], [0.03780742734670639], [0.23160281777381897], [0.18210230767726898], [0.21688763797283173], [0.009429873898625374], [0.23962129652500153]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.39208778738975525], [0.2094176560640335], [0.1283661276102066], [0.27551141381263733], [0.03479340672492981], [0.3570266664028168], [0.11812948435544968], [0.2456916868686676], [0.48227444291114807]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f9a778069b418f7d6f8803c18b3921f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39371946454048157], [0.050644610077142715], [0.09682576358318329], [0.37938541173934937], [0.37218934297561646], [0.31943804025650024], [0.009334404021501541], [0.027887742966413498], [0.1417359560728073]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.49722179770469666], [0.4310191571712494], [0.3834795355796814], [0.30722612142562866], [0.2178652435541153], [0.11710164695978165], [0.23001685738563538], [0.22062554955482483], [0.498458594083786]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4d62f21a69b8d75a4b3559a2e7458fd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08568544685840607], [0.1383717805147171], [0.2306109517812729], [0.03215041384100914], [0.05588645488023758], [0.1058163195848465], [0.12910811603069305], [0.02052994631230831], [0.24312002956867218]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.1547079086303711], [0.3591558635234833], [0.15294159948825836], [0.019754718989133835], [0.2135521024465561], [0.2554653286933899], [0.3142800033092499], [0.4077095687389374], [0.42899003624916077]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dfcef0fef8fab4a05f88bbd205b63c07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08652636408805847], [0.38375985622406006], [0.22444428503513336], [0.3983890116214752], [0.28843000531196594], [0.07827463001012802], [0.3716101050376892], [0.20676524937152863], [0.260421484708786]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.39656439423561096], [0.12202204018831253], [0.0054341829381883144], [0.014495838433504105], [0.0013494952581822872], [0.28443190455436707], [0.2855451703071594], [0.20876038074493408], [0.38022154569625854]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6ab8d4e15efa4ab6685b1d4b16a9055c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.023871352896094322], [0.0072302743792533875], [0.028566552326083183], [-0.012393927201628685], [-0.014890296384692192], [-0.004542335867881775], [-0.037730999290943146], [0.046309053897857666], [0.10882711410522461]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_57300b75399eb3a297d1b5ae346d5b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36820653080940247], [0.1383717805147171], [0.2306109517812729], [0.03780742734670639], [0.23160281777381897], [0.18210230767726898], [0.21688763797283173], [0.02052994631230831], [0.24312002956867218]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.1547079086303711], [0.2094176560640335], [0.1283661276102066], [0.019754718989133835], [0.03479340672492981], [0.2554653286933899], [0.11812948435544968], [0.2456916868686676], [0.42899003624916077]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e99dda31b438944e774c2dd8272ee17a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39371946454048157], [0.38375985622406006], [0.22444428503513336], [0.3983890116214752], [0.37218934297561646], [0.31943804025650024], [0.3716101050376892], [0.20676524937152863], [0.260421484708786]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.39656439423561096], [0.12202204018831253], [0.0054341829381883144], [0.014495838433504105], [0.0013494952581822872], [0.11710164695978165], [0.23001685738563538], [0.20876038074493408], [0.38022154569625854]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_473ed62fcf0ad6ec79b1600d60c809ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.000607388443313539], [-0.018595393747091293], [0.022392649203538895], [0.00693031121045351], [0.07298476994037628], [-0.014844009652733803], [0.013983488082885742], [0.00044922734377905726], [0.022267237305641174]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.023871352896094322], [0.0072302743792533875], [0.028566552326083183], [-0.012393927201628685], [-0.014890296384692192], [-0.004542335867881775], [-0.037730999290943146], [0.046309053897857666], [0.10882711410522461]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_abcce45492f97a3cf126b8f399949a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [-0.0], [-0.0], [-0.0], [-0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[40.3016242980957], [1.388820767402649], [-0.27571114897727966], [2.788365364074707], [1.20401930809021], [0.6939953565597534], [3.698253870010376], [-102.08600616455078], [-3.887320041656494]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b2649627c56cc22f68d3cbea932371b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_acb2b446c113941f8ae47e1572ee95c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.06146731972694397]], [[0.38457757234573364]], [[0.1499379724264145]], [[0.2519291639328003]], [[0.4287847876548767]], [[0.48430103063583374]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.7040383219718933]], [[0.5041806101799011]], [[0.5248106718063354]], [[0.5276604890823364]], [[0.8099044561386108]], [[0.5424766540527344]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e307239dd03430cb7079aef3b0aa6f2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.03920156881213188]], [[0.022001300007104874]], [[0.24809589982032776]], [[0.39565593004226685]], [[0.10346271097660065]], [[0.42565131187438965]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.7184906005859375]], [[0.5257463455200195]], [[0.6536774635314941]], [[0.6953741312026978]], [[0.7265492677688599]], [[0.5624583959579468]]], dtype='float32').reshape([6, 1, 1]),
        ]



class PrimitiveOp_ef457a8c2827717ad98c98ab5f8f38cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_14bae7859044d7ac7a1e855f2042d793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef457a8c2827717ad98c98ab5f8f38cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dce5143f377b8806b1983e2a0455ea5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37b14280938c93f8570b3df785eeb1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37b14280938c93f8570b3df785eeb1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37b14280938c93f8570b3df785eeb1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37b14280938c93f8570b3df785eeb1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37b14280938c93f8570b3df785eeb1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37b14280938c93f8570b3df785eeb1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37b14280938c93f8570b3df785eeb1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37b14280938c93f8570b3df785eeb1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37b14280938c93f8570b3df785eeb1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37b14280938c93f8570b3df785eeb1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37b14280938c93f8570b3df785eeb1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03b1906bab8cf1f48b4606ade5191acd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f0222eca32e6403f4049a2fb3b688ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dce5143f377b8806b1983e2a0455ea5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2ee5247c11c6647ab647c3ba412596ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13856299221515656, 0.4353436827659607, 0.2094043344259262, 0.2879944443702698], [0.08696585893630981, 0.2929264307022095, 0.17700904607772827, 0.2517296075820923], [0.18547385931015015, 0.4464535713195801, 0.4669950008392334, 0.4609496593475342], [0.08696585893630981, 0.2929264307022095, 0.17700904607772827, 0.2517296075820923], [0.18547385931015015, 0.4464535713195801, 0.4669950008392334, 0.4609496593475342], [0.04176447167992592, 0.43585583567619324, 0.007390841841697693, 0.41927239298820496], [0.04176447167992592, 0.43585583567619324, 0.007390841841697693, 0.41927239298820496]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.3110031485557556, 0.12883557379245758, 0.014142144471406937, 0.29536378383636475], [0.03524753451347351, 0.44889235496520996, 0.32588690519332886, 0.4541289508342743], [0.3971027433872223, 0.4102397561073303, 0.28529155254364014, 0.39603105187416077], [0.03524753451347351, 0.44889235496520996, 0.32588690519332886, 0.4541289508342743], [0.3971027433872223, 0.4102397561073303, 0.28529155254364014, 0.39603105187416077], [0.43009355664253235, 0.49960651993751526, 0.3261641263961792, 0.4162203073501587], [0.43009355664253235, 0.49960651993751526, 0.3261641263961792, 0.4162203073501587]], dtype='float32').reshape([7, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e93be1b02e7657a17ee401eb2ea5cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e93be1b02e7657a17ee401eb2ea5cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6a0d88c3b31ac0ec78be6d8038ebdb82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_18b385f2773e20549fa68a071089e091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8b8646482c76ef0cc3a8840014a75d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2608264982700348, 0.4725295603275299, 0.30785173177719116, 0.08038189262151718, 0.0575774721801281, 0.3921468257904053], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1027945876121521, 0.3881237208843231, 0.152043879032135, 0.21335071325302124, 0.38008421659469604, 0.12074317038059235], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ae96253cfd7568e986d861da85f3af2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4957810342311859, 0.08107513189315796, 0.33436471223831177, 0.4922436773777008, 0.36378422379493713, 0.13394984602928162], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2527327835559845, 0.1542942374944687, 0.470980703830719, 0.32545068860054016, 0.27355799078941345, 0.35376250743865967], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_54ff2d0396fef3ecaa5346e0d5811295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24978351593017578, 0.11446373909711838, 0.06451474130153656, 0.04623079672455788, 0.3613818883895874, 0.4987070560455322], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2304920107126236, 0.0650036409497261, 0.457919716835022, 0.030212104320526123, 0.35011738538742065, 0.12877273559570312], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8fc350a0642ca1eee42599eb3bcda7e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2845926582813263, 0.16627027094364166, 0.25496557354927063, 0.29948753118515015, 0.21094956994056702, 0.2771330177783966], dtype='float32').reshape([6]),
            paddle.to_tensor([0.23068910837173462, 0.1622818112373352, 0.3164260983467102, 0.16976673901081085, 0.15035927295684814, 0.22009573876857758], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_989f2114979a7884f0b0ca404ba695ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24978351593017578, 0.11446373909711838, 0.06451474130153656, 0.04623079672455788, 0.3613818883895874, 0.3921468257904053], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2304920107126236, 0.3881237208843231, 0.457919716835022, 0.21335071325302124, 0.38008421659469604, 0.12877273559570312], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8cff451749268b28a898dd3399de8eb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2845926582813263, 0.1542942374944687, 0.25496557354927063, 0.29948753118515015, 0.21094956994056702, 0.2771330177783966], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2527327835559845, 0.1622818112373352, 0.470980703830719, 0.32545068860054016, 0.27355799078941345, 0.35376250743865967], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c4cefd8604ec773bae6ee8cd8740d8db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2608264982700348, 0.4725295603275299, 0.30785173177719116, 0.21335071325302124, 0.38008421659469604, 0.3921468257904053], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1027945876121521, 0.3881237208843231, 0.152043879032135, 0.21335071325302124, 0.38008421659469604, 0.12074317038059235], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ade7d3f0ac72a7e36b62e89bbd6b2cc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4957810342311859, 0.1542942374944687, 0.470980703830719, 0.4922436773777008, 0.36378422379493713, 0.35376250743865967], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2527327835559845, 0.1542942374944687, 0.470980703830719, 0.32545068860054016, 0.27355799078941345, 0.35376250743865967], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b31f71a1dc3b2bd3a84e6516c438c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.039449259638786316, 0.00019726960454136133, 0.024178875610232353, 0.002077957382425666, 0.0006825195741839707, 0.02110004797577858], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0006146249361336231, 0.0, 0.0, 0.0, 0.0, -0.0], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52d3eee2ce8e716ae1212a9dea1b69a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18181054294109344, 0.4303266406059265, 0.22994780540466309, 0.1468663066625595, 0.21883083879947662, 0.2564449906349182], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2401377558708191, 0.08973369002342224, 0.26121723651885986, 0.03822144865989685, 0.35574963688850403, 0.3137398958206177], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_239165a78e3bbb76d225e4f3a0cd1477(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3742569088935852, 0.11768468469381332, 0.4026727080345154, 0.4088471829891205, 0.3186711072921753, 0.24385617673397064], dtype='float32').reshape([6]),
            paddle.to_tensor([0.25764089822769165, 0.16427603363990784, 0.2856958508491516, 0.2346271276473999, 0.18065442144870758, 0.2486143708229065], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_64c1ad6254405a9fadac3cd392be9c4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2608264982700348, 0.4725295603275299, 0.30785173177719116, 0.21335071325302124, 0.38008421659469604, 0.4987070560455322], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1027945876121521, 0.0650036409497261, 0.152043879032135, 0.030212104320526123, 0.35011738538742065, 0.12074317038059235], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9ce8b4d942fb0b384c967fa1d9e3c3f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4957810342311859, 0.16627027094364166, 0.470980703830719, 0.4922436773777008, 0.36378422379493713, 0.35376250743865967], dtype='float32').reshape([6]),
            paddle.to_tensor([0.23068910837173462, 0.1542942374944687, 0.3164260983467102, 0.16976673901081085, 0.15035927295684814, 0.22009573876857758], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c1be02e8773f94c2bee9eb11fee73ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3436858057975769, 1.4903305768966675, 1.41582190990448, 0.12286394089460373, 0.18381407856941223, 1.4178187847137451], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5765213966369629, -0.856249988079071, -0.8509345054626465, -0.6730365753173828, -1.298005223274231, -0.8900416493415833], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_57e2653570c78928800aab0f941d01e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d76594d7bc73a01590ff4d5f569ebf93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d76594d7bc73a01590ff4d5f569ebf93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d76594d7bc73a01590ff4d5f569ebf93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d76594d7bc73a01590ff4d5f569ebf93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d76594d7bc73a01590ff4d5f569ebf93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d76594d7bc73a01590ff4d5f569ebf93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d76594d7bc73a01590ff4d5f569ebf93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d76594d7bc73a01590ff4d5f569ebf93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d76594d7bc73a01590ff4d5f569ebf93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d76594d7bc73a01590ff4d5f569ebf93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d76594d7bc73a01590ff4d5f569ebf93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52d6c0904952e02947e994d334c88237(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38dc692e168c69c6d9272c552d7727e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_57e2653570c78928800aab0f941d01e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_9fec7ada1680728cff3c134be87b6aa2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8400, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0bd7d880fc7306d155fe38898579c30a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fec7ada1680728cff3c134be87b6aa2
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_69cefdcc79ddad326fae6984d3c01b5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([0.16727307438850403, 0.29837217926979065, 0.19894284009933472, 0.48071572184562683, 0.3676881194114685, 0.4198280870914459, 0.17023754119873047, 0.16682079434394836, 0.41211000084877014, 0.27952146530151367, 0.03478027507662773, 0.25146594643592834, 0.3347516655921936, 0.07699854671955109, 0.16147416830062866, 0.18425174057483673, 0.1390560418367386, 0.17906422913074493, 0.1935458481311798, 0.2319454848766327, 0.13256007432937622, 0.057637352496385574, 0.14197881519794464, 0.27830740809440613], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8d5c4ef7476afda985a09d0fd3bd5879(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.16727307438850403, 0.29837217926979065, 0.19894284009933472, 0.48071572184562683, 0.3676881194114685, 0.4198280870914459, 0.17023754119873047, 0.16682079434394836, 0.41211000084877014, 0.27952146530151367, 0.03478027507662773, 0.25146594643592834, 0.3347516655921936, 0.07699854671955109, 0.16147416830062866, 0.18425174057483673, 0.1390560418367386, 0.17906422913074493, 0.1935458481311798, 0.2319454848766327, 0.13256007432937622, 0.057637352496385574, 0.14197881519794464, 0.27830740809440613], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f4ad3e30e2f4714495615515f0ad398(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb30f13e2bbdd96dfc91a7c85e7d5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb30f13e2bbdd96dfc91a7c85e7d5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb30f13e2bbdd96dfc91a7c85e7d5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb30f13e2bbdd96dfc91a7c85e7d5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb30f13e2bbdd96dfc91a7c85e7d5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb30f13e2bbdd96dfc91a7c85e7d5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb30f13e2bbdd96dfc91a7c85e7d5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb30f13e2bbdd96dfc91a7c85e7d5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb30f13e2bbdd96dfc91a7c85e7d5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb30f13e2bbdd96dfc91a7c85e7d5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb30f13e2bbdd96dfc91a7c85e7d5d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3ed75efcd84ec090d0b5f8a4bdcffa95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1d0e3e7e00154ab42eff6103c79fdb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f4ad3e30e2f4714495615515f0ad398(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a4e823eb61d9b61aba09226acd691f40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
            paddle.to_tensor([0.09576865285634995, 0.21895988285541534, 0.47236356139183044, 0.28012216091156006], dtype='float32').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bb686935ff7497e52ed3e1c147aa8a15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09576865285634995, 0.21895988285541534, 0.47236356139183044, 0.28012216091156006], dtype='float32').reshape([4]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([4]),
        ]



class PrimitiveOp_1917ea0a108920c4069ffa6dfbed0f79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8cdd6eeef482a5f5c2da614c8ee3b44b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1917ea0a108920c4069ffa6dfbed0f79
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int32').reshape([1]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_56c4a2e44901467a3e8ec6b3b0b1c5e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1917ea0a108920c4069ffa6dfbed0f79
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d9a63d5e912ab50e13dd8f29dcfd2c40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.48869359493255615, 0.25505194067955017, 0.2965860962867737, 0.1435116082429886], [0.29652026295661926, 0.17479072511196136, 0.12994545698165894, 0.3918682336807251], [0.03457598015666008, 0.21453428268432617, 0.40904825925827026, 0.13430796563625336], [0.4367668628692627, 0.10261180251836777, 0.21258141100406647, 0.15890343487262726], [0.4367668628692627, 0.10261180251836777, 0.21258141100406647, 0.15890343487262726], [0.03457598015666008, 0.21453428268432617, 0.40904825925827026, 0.13430796563625336]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.17786620557308197, 0.3837328553199768, 0.4018065929412842, 0.052461110055446625], [0.1788792461156845, 0.17232021689414978, 0.44501304626464844, 0.18641409277915955], [0.23814435303211212, 0.0061455052345991135, 0.03816644474864006, 0.24216414988040924], [0.18565472960472107, 0.4981105327606201, 0.3184441328048706, 0.44102829694747925], [0.18565472960472107, 0.4981105327606201, 0.3184441328048706, 0.44102829694747925], [0.23814435303211212, 0.0061455052345991135, 0.03816644474864006, 0.24216414988040924]], dtype='float32').reshape([6, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a5ae8a733e11dc71ae541a385555c8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.31454479694366455, 0.10439124703407288, 0.16489987075328827, 0.18370254337787628], [0.11546322703361511, 0.26400521397590637, 0.06056874617934227, 0.09392143040895462], [0.4271167516708374, 0.453997939825058, 0.2668359875679016, 0.30915793776512146], [0.05479086562991142, 0.4311014413833618, 0.2002004086971283, 0.21852238476276398], [0.31454479694366455, 0.10439124703407288, 0.16489987075328827, 0.18370254337787628]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.26386094093322754, 0.27112436294555664, 0.39497309923171997, 0.42580345273017883], [0.47162744402885437, 0.0874064713716507, 0.2927992045879364, 0.4740701913833618], [0.15716779232025146, 0.0017121427226811647, 0.13402105867862701, 0.2784697711467743], [0.2673611044883728, 0.20431914925575256, 0.4392146170139313, 0.18085090816020966], [0.26386094093322754, 0.27112436294555664, 0.39497309923171997, 0.42580345273017883]], dtype='float32').reshape([5, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_56bd3d7f1023b32564a39a9793b4d622(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1cbcd7991ed6fdb0c9c5b0ce675977d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11903903633356094]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3555246889591217]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d41f96ee1b152bb09b6faa5f12d868b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0831770971417427]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.45340871810913086]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05f1cc7c1bf89a05d469eb24c2730477(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33794644474983215]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2977246940135956]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ad03b962229cddadf6f2e39118379806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12983585894107819]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.45340871810913086]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1cbcd7991ed6fdb0c9c5b0ce675977d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11903903633356094]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3555246889591217]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a06e94cca90c7caf756fbb7ab054eaab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0831770971417427]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.43772584199905396]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9553c98238d3d876f3f0e0fe7f49f6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07083103060722351]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05f1cc7c1bf89a05d469eb24c2730477(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33794644474983215]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2977246940135956]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a0b8984f934abc666cf9a570906a85b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12983585894107819]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.43772584199905396]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6e566f0e9601517bfacd01266c4611de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.012383874505758286]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.07083103060722351]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_594049bfdd18338219bc9314c2f732df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[6.719618320465088]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5238a0ed6b632fd19c88e5f35d3f4b47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20998290181159973], [0.18442489206790924], [0.1968512237071991], [0.20288529992103577], [0.21599259972572327], [0.19651634991168976]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3895230293273926], [0.4666270613670349], [0.17671829462051392], [0.3581591248512268], [0.42263373732566833], [0.24344341456890106]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_792a83a8650844cd06cca95700a8fbf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4151418209075928], [0.2509276866912842], [0.13929075002670288], [0.03818127512931824], [0.08567338436841965], [0.23274247348308563]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.4103717803955078], [0.39778658747673035], [0.40150901675224304], [0.42183321714401245], [0.1773831695318222], [0.48681026697158813]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_789850f723c6754bce6385edefe0a3f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20998290181159973], [0.4498758018016815], [0.24683597683906555], [0.20288529992103577], [0.21599259972572327], [0.19651634991168976]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.15491808950901031], [0.16919776797294617], [0.1292070597410202], [0.12714418768882751], [0.33482757210731506], [0.2098187357187271]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dc789da2137c74034b4363355f450b61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4901370406150818], [0.2509276866912842], [0.2429288774728775], [0.06132770702242851], [0.08567338436841965], [0.23274247348308563]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.4103717803955078], [0.39778658747673035], [0.40150901675224304], [0.42183321714401245], [0.09941722452640533], [0.25498807430267334]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_306651a096272fa6d733b41ea8025470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22682112455368042], [0.18442489206790924], [0.1968512237071991], [0.46012169122695923], [0.40794169902801514], [0.4862612187862396]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3895230293273926], [0.4666270613670349], [0.17671829462051392], [0.3581591248512268], [0.42263373732566833], [0.24344341456890106]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c77c433ea911e2411aac58d395d8556e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4151418209075928], [0.3606884777545929], [0.13929075002670288], [0.03818127512931824], [0.28175589442253113], [0.3325986862182617]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.1010279506444931], [0.2909882366657257], [0.2298392504453659], [0.29276591539382935], [0.1773831695318222], [0.48681026697158813]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1606ef2561d5873d54f91490159d98b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0467146635055542], [-0.060889631509780884], [-0.02047661691904068], [-0.05326319485902786], [9.980075992643833e-05], [-0.037149399518966675]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6f8b324f7e7b0868e1c9528cbc26d0cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22682112455368042], [0.4498758018016815], [0.24683597683906555], [0.46012169122695923], [0.40794169902801514], [0.4862612187862396]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.15491808950901031], [0.16919776797294617], [0.1292070597410202], [0.12714418768882751], [0.33482757210731506], [0.2098187357187271]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ecb901fe7978d0cea63d57c249161309(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4901370406150818], [0.3606884777545929], [0.2429288774728775], [0.06132770702242851], [0.28175589442253113], [0.3325986862182617]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.1010279506444931], [0.2909882366657257], [0.2298392504453659], [0.29276591539382935], [0.09941722452640533], [0.25498807430267334]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_264093c99d91c0e58506e35590b0d9ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.02797812409698963], [0.019563326612114906], [0.0015397188253700733], [-0.07706371694803238], [0.01333153247833252], [0.021454868838191032]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.0467146635055542], [-0.060889631509780884], [-0.02047661691904068], [-0.05326319485902786], [9.980086178984493e-05], [-0.037149399518966675]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0a42593a2c026c8062f5eede8020720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [-0.0], [-0.0], [0.0], [-0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[2.6696853637695312], [4.1124372482299805], [14.298932075500488], [0.30884212255477905], [0.9925139546394348], [2.731513738632202]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc98259315ae6586ac9df6a85b6c5562(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1824364960193634, 0.11383223533630371, 0.15641945600509644, 0.39408645033836365], [0.44869938492774963, 0.021236063912510872, 0.01257316768169403, 0.1094929650425911], [0.23305965960025787, 0.25187548995018005, 0.1086021214723587, 0.40324482321739197], [0.4280799329280853, 0.38631755113601685, 0.44555965065956116, 0.33409902453422546]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([[0.1865062266588211, 0.10327757894992828, 0.31373849511146545, 0.37114739418029785], [0.4478083550930023, 0.3082262873649597, 0.22778934240341187, 0.13071022927761078], [0.07917411625385284, 0.3322516679763794, 0.15485522150993347, 0.46097493171691895], [0.12284959852695465, 0.0587301105260849, 0.26185959577560425, 0.17734073102474213]], dtype='float32').reshape([4, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ad01ba209aa563c77d8a0f5ecf6c7ca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3c2c2cf2a99e2e53e9cb3104d1d7037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c250574f3749206292125dd9ec423b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c250574f3749206292125dd9ec423b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c250574f3749206292125dd9ec423b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c250574f3749206292125dd9ec423b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c250574f3749206292125dd9ec423b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c250574f3749206292125dd9ec423b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c250574f3749206292125dd9ec423b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c250574f3749206292125dd9ec423b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c250574f3749206292125dd9ec423b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c250574f3749206292125dd9ec423b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c250574f3749206292125dd9ec423b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15aa17eea687877cf988c12a2c059c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_358bc8c1eb98b6e964028dcc0f56dec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3c2c2cf2a99e2e53e9cb3104d1d7037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a9ea8ff0bae1a32aa2249436f11ffaaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.21805590391159058, 0.369320809841156, 0.46295464038848877, 0.32399559020996094], [0.21805590391159058, 0.369320809841156, 0.46295464038848877, 0.32399559020996094], [0.45063576102256775, 0.08473407477140427, 0.2964290678501129, 0.17995186150074005], [0.35198330879211426, 0.0768512561917305, 0.0005001812824048102, 0.43629372119903564], [0.23088055849075317, 0.3435775935649872, 0.022802097722887993, 0.006141966208815575], [0.1841537058353424, 0.24625396728515625, 0.19545932114124298, 0.41149935126304626], [0.0622473768889904, 0.010412028059363365, 0.32384076714515686, 0.019491543993353844]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.11718153953552246, 0.18766602873802185, 0.3853262960910797, 0.17273500561714172], [0.11718153953552246, 0.18766602873802185, 0.3853262960910797, 0.17273500561714172], [0.11498585343360901, 0.39520618319511414, 0.019509049132466316, 0.21388070285320282], [0.11598620563745499, 0.13958659768104553, 0.2296896129846573, 0.40918001532554626], [0.08412293344736099, 0.36157235503196716, 0.21181900799274445, 0.0409696027636528], [0.4178256392478943, 0.11422522366046906, 0.3132604658603668, 0.10981956124305725], [0.3702118396759033, 0.0816604271531105, 0.2898832857608795, 0.05852610245347023]], dtype='float32').reshape([7, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40f637a1c203bb54d195c3251937c47a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_19bcde026d279fc2319de30a93e14494(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef457a8c2827717ad98c98ab5f8f38cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e16d1298a6a509575e0c4f4d15de3a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec11672a6152cbe658064c630cc61cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec11672a6152cbe658064c630cc61cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec11672a6152cbe658064c630cc61cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec11672a6152cbe658064c630cc61cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec11672a6152cbe658064c630cc61cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec11672a6152cbe658064c630cc61cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec11672a6152cbe658064c630cc61cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec11672a6152cbe658064c630cc61cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec11672a6152cbe658064c630cc61cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec11672a6152cbe658064c630cc61cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec11672a6152cbe658064c630cc61cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fcb8107f4be5eea7babae7f8cd89696e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f56d38e40cb258115066d09088cf23a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e16d1298a6a509575e0c4f4d15de3a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_437575c72e6763468f38fbb580627186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b62a7e7d7aeb8dfe79317e11a31487f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b62a7e7d7aeb8dfe79317e11a31487f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b62a7e7d7aeb8dfe79317e11a31487f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b62a7e7d7aeb8dfe79317e11a31487f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b62a7e7d7aeb8dfe79317e11a31487f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b62a7e7d7aeb8dfe79317e11a31487f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b62a7e7d7aeb8dfe79317e11a31487f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b62a7e7d7aeb8dfe79317e11a31487f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b62a7e7d7aeb8dfe79317e11a31487f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b62a7e7d7aeb8dfe79317e11a31487f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b62a7e7d7aeb8dfe79317e11a31487f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_21a50b949a46359c33f1a60837b3d774(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67dd1ecbca2884ee78c7cd1a7eea2215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_437575c72e6763468f38fbb580627186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_515f52247eb0925eb57e78c260d58f96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3a19ffbad53f86cf6124faf17bca69f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1983117163181305, 0.4262564480304718, 0.44021251797676086, 0.45081812143325806], [0.3675566613674164, 0.2470126450061798, 0.4268859326839447, 0.33921435475349426], [0.3675566613674164, 0.2470126450061798, 0.4268859326839447, 0.33921435475349426], [0.33047568798065186, 0.14772096276283264, 0.16170357167720795, 0.1762121468782425], [0.1535663604736328, 0.11277178674936295, 0.3490486741065979, 0.19976338744163513], [0.052271824330091476, 0.46588462591171265, 0.35674288868904114, 0.013300606049597263]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.4848327338695526, 0.167837992310524, 0.48880672454833984, 0.4181344211101532], [0.32352542877197266, 0.27273812890052795, 0.30801448225975037, 0.2090080976486206], [0.32352542877197266, 0.27273812890052795, 0.30801448225975037, 0.2090080976486206], [0.005183657631278038, 0.49299538135528564, 0.12437743693590164, 0.14061975479125977], [0.10720117390155792, 0.40686947107315063, 0.33792147040367126, 0.4700213372707367], [0.44660133123397827, 0.26956701278686523, 0.28133127093315125, 0.14439621567726135]], dtype='float32').reshape([6, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_293d93bbcf6e674049780843ff641a48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_32469e7c3a76900c5c34599e508cf549(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 1, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c0273208412600b6eafd8a44aa9f1e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32469e7c3a76900c5c34599e508cf549
    def get_inputs(self):
        return [
            paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.39817842841148376, 12.484559059143066, 2.655482292175293, 1.6678518056869507], [2.003458261489868, 1.472330927848816, 0.9269778728485107, 0.15994922816753387]], dtype='float32').reshape([2, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_8b891c3c0afc2d4183664ce45c3a7b49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6069, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea82364bb78e0e4eedc177c19e5452e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b891c3c0afc2d4183664ce45c3a7b49
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5506a7d5add76bd5c93be0c3ec95da8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_af27d48e9942876178a44f8989dc4d03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 1, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d626780282058141aa20d73283d3ee6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af27d48e9942876178a44f8989dc4d03
    def get_inputs(self):
        return [
            paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1.776720643043518, 0.14957483112812042, 0.3515414893627167, 1.572020173072815], [4.945065021514893, 5.548974990844727, 1.1182193756103516, 0.07188649475574493]], dtype='float32').reshape([2, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6e064d47fc2ef1aad541cfaac0081039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27102693915367126], [0.21676185727119446], [0.3153221905231476], [0.01315593346953392], [0.1262916475534439]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4706946909427643], [0.39319300651550293], [0.19910383224487305], [0.3212777376174927], [0.29648688435554504]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cac090a0c6006c4ff68fabaed33f11a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39167019724845886], [0.046127624809741974], [0.04106692224740982], [0.014858669601380825], [0.17310704290866852]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.16876435279846191], [0.1894516497850418], [0.3004133403301239], [0.4173230230808258], [0.3782731890678406]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68f0d019e4c9fd642d19ba5994ae6632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3077138066291809], [0.21676185727119446], [0.36836960911750793], [0.01315593346953392], [0.1262916475534439]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.3776218891143799], [0.017230553552508354], [0.12865380942821503], [0.0789160430431366], [0.29648688435554504]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46a3ac494b39c23c2c6c2ad758ea68ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39167019724845886], [0.046127624809741974], [0.13554736971855164], [0.2480829954147339], [0.17310704290866852]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.04191706329584122], [0.1894516497850418], [0.11870089173316956], [0.0004433089052326977], [0.28320610523223877]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77363bca5aab326a7078b852be665c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27102693915367126], [0.3185052275657654], [0.3153221905231476], [0.430990993976593], [0.4189223051071167]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4706946909427643], [0.39319300651550293], [0.19910383224487305], [0.3212777376174927], [0.27562597393989563]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_16a674fb86173a0a7b96accf14c24b02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4086301028728485], [0.391907662153244], [0.04106692224740982], [0.014858669601380825], [0.4931931793689728]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.16876435279846191], [0.1280040442943573], [0.3004133403301239], [0.4173230230808258], [0.3782731890678406]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_26701ed506c04a30ea9dbe81ee2556d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.07234402745962143], [-0.048308003693819046], [-0.026102447882294655], [-0.060440488159656525], [0.03520594909787178]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3c14aeb1c83e16814c4b64af6530f2a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3077138066291809], [0.3185052275657654], [0.36836960911750793], [0.430990993976593], [0.4189223051071167]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.3776218891143799], [0.017230553552508354], [0.12865380942821503], [0.0789160430431366], [0.27562597393989563]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15432243cbc4d9fe3899ec0e1153673d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4086301028728485], [0.391907662153244], [0.13554736971855164], [0.2480829954147339], [0.4931931793689728]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.04191706329584122], [0.1280040442943573], [0.11870089173316956], [0.0004433089052326977], [0.28320610523223877]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32464289f44b3279d82a52003896bdb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.025636205449700356], [0.07950747758150101], [0.0040383669547736645], [0.08718772977590561], [0.03009037673473358]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.07234402745962143], [-0.048308003693819046], [-0.026102447882294655], [-0.060440488159656525], [0.03520594909787178]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6aaa08ff2c87645dd7e343b5022045c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-1.821947693824768], [1.6075907945632935], [7.4636149406433105], [1.6932224035263062], [-0.1700069159269333]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f4eed4ac0bd1910c2ef37732dde94533(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef457a8c2827717ad98c98ab5f8f38cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df1fa1f7cbd3e8ff8fbaf296352cf42c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5360d447777f71181e9df638492f3aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5360d447777f71181e9df638492f3aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5360d447777f71181e9df638492f3aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5360d447777f71181e9df638492f3aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5360d447777f71181e9df638492f3aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5360d447777f71181e9df638492f3aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5360d447777f71181e9df638492f3aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5360d447777f71181e9df638492f3aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5360d447777f71181e9df638492f3aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5360d447777f71181e9df638492f3aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5360d447777f71181e9df638492f3aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0cf1b79d6b0bda99fe30897e159b4ad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_482a57e3f8a9d1754788fd87985c8f0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df1fa1f7cbd3e8ff8fbaf296352cf42c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df4cb3e0133b5df466a7321a93627e00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1be842004c1ded63014828f8c93c5ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1be842004c1ded63014828f8c93c5ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1be842004c1ded63014828f8c93c5ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1be842004c1ded63014828f8c93c5ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1be842004c1ded63014828f8c93c5ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1be842004c1ded63014828f8c93c5ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1be842004c1ded63014828f8c93c5ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1be842004c1ded63014828f8c93c5ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1be842004c1ded63014828f8c93c5ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1be842004c1ded63014828f8c93c5ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1be842004c1ded63014828f8c93c5ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8d5f52ddf8383c305d82d62c38a7f453(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc25a83b16b04b6ad5a3de78a8e76b5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df4cb3e0133b5df466a7321a93627e00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_10117e11107ef9c9838ff68488843148(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f67919162c1f60ec850e0ef57ffb2e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f67919162c1f60ec850e0ef57ffb2e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f67919162c1f60ec850e0ef57ffb2e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f67919162c1f60ec850e0ef57ffb2e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f67919162c1f60ec850e0ef57ffb2e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f67919162c1f60ec850e0ef57ffb2e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f67919162c1f60ec850e0ef57ffb2e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f67919162c1f60ec850e0ef57ffb2e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f67919162c1f60ec850e0ef57ffb2e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f67919162c1f60ec850e0ef57ffb2e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f67919162c1f60ec850e0ef57ffb2e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dba09ba03ffefda8f2726c71385338fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9f8e1c85db4557fff30430af9b30c7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_10117e11107ef9c9838ff68488843148(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5eb26b92970518f687a0bea510e387ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef457a8c2827717ad98c98ab5f8f38cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3a508b496778ed0b643bc983d4abc57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_654c5f6af86ec6b7bca9d326c0167e14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 512, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_033ed509de49dffc0537d194bb79278b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_654c5f6af86ec6b7bca9d326c0167e14
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4bd2d697bf6182f52895306cc885df7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([0.4473949074745178, 0.08388441801071167, 0.17490623891353607, 0.2467467337846756, 0.14741015434265137, 0.14820529520511627, 0.4406585693359375, 0.37450501322746277, 0.04052567854523659, 0.48126161098480225, 0.4376707375049591, 0.2969852685928345, 0.2029794454574585, 0.15935193002223969, 0.2048863172531128, 0.25172749161720276, 0.38391077518463135, 0.35427677631378174, 0.34242749214172363, 0.49203726649284363], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_09380d0d6e063af8224ede249c2f0ebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4473949074745178, 0.08388441801071167, 0.17490623891353607, 0.2467467337846756, 0.14741015434265137, 0.14820529520511627, 0.4406585693359375, 0.37450501322746277, 0.04052567854523659, 0.48126161098480225, 0.4376707375049591, 0.2969852685928345, 0.2029794454574585, 0.15935193002223969, 0.2048863172531128, 0.25172749161720276, 0.38391077518463135, 0.35427677631378174, 0.34242749214172363, 0.49203726649284363], dtype='float32').reshape([20]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_01bb10db17854f1b70da84dea0683b7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1317010074853897], [0.1832035928964615], [0.005624867510050535], [0.382903516292572]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4710065722465515], [0.27305787801742554], [0.37650251388549805], [0.3584076464176178]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1a0b2eb921f5a565e0e6ec6b8f868f8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2849007844924927], [0.1524905115365982], [0.36212438344955444], [0.4171924889087677]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3558596074581146], [0.34534937143325806], [0.45495909452438354], [0.43209323287010193]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b057d51f5ce2a7ef4b1604848a7e43f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1317010074853897], [0.1832035928964615], [0.005624867510050535], [0.382903516292572]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4710065722465515], [0.27305787801742554], [0.1443967968225479], [0.18359437584877014]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f54ee33ad5572909cc204d2f34c47ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.407511442899704], [0.41122034192085266], [0.36212438344955444], [0.47452500462532043]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3558596074581146], [0.24631327390670776], [0.38240188360214233], [0.43209323287010193]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8757f20916feccccae702f6990bca23f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.21841885149478912], [0.4341205954551697], [0.17934243381023407], [0.48898959159851074]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.11861217021942139], [0.1668710708618164], [0.37650251388549805], [0.3584076464176178]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dc8042352da3be90258765fe14bec82f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2849007844924927], [0.1524905115365982], [0.386353075504303], [0.4171924889087677]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.17836087942123413], [0.34534937143325806], [0.45495909452438354], [0.29443272948265076]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62106d4632ce0001929f0696891a2a61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.006892362609505653], [-0.06635904312133789], [0.016340315341949463], [0.024487247690558434]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47d77aaf788cd9fc4f442a835c572b03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.21841885149478912], [0.4341205954551697], [0.17934243381023407], [0.48898959159851074]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.11861217021942139], [0.1668710708618164], [0.1443967968225479], [0.18359437584877014]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f9ba8564672e52eccfb9552453f64491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.407511442899704], [0.41122034192085266], [0.386353075504303], [0.47452500462532043]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.17836087942123413], [0.24631327390670776], [0.38240188360214233], [0.29443272948265076]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb8ae06b21b230a17f188b9522134849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.022870756685733795], [0.04407133534550667], [0.00013807702634949237], [0.05499931797385216]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.006892362609505653], [-0.06635904312133789], [0.016340315341949463], [0.024487247690558434]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6691748a8a87d09f2a04ccd61d787fb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[1.301361322402954], [2.505718946456909], [-117.34202575683594], [0.5547717809677124]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2bdf48ae6973d8a4487566beb1fae511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c2dca27c7255126b7a4f8c21d919d4ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03fa676653657fea35ebb0dd3545f434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03fa676653657fea35ebb0dd3545f434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03fa676653657fea35ebb0dd3545f434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03fa676653657fea35ebb0dd3545f434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03fa676653657fea35ebb0dd3545f434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03fa676653657fea35ebb0dd3545f434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03fa676653657fea35ebb0dd3545f434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03fa676653657fea35ebb0dd3545f434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03fa676653657fea35ebb0dd3545f434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03fa676653657fea35ebb0dd3545f434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03fa676653657fea35ebb0dd3545f434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15aa17eea687877cf988c12a2c059c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_358bc8c1eb98b6e964028dcc0f56dec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c2dca27c7255126b7a4f8c21d919d4ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_033ed509de49dffc0537d194bb79278b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_654c5f6af86ec6b7bca9d326c0167e14
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aae057c2bbb4508044ffb10c39a7e238(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef457a8c2827717ad98c98ab5f8f38cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_7182d24cb9ed83c3e793bde3926531e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6804, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68b802329411cbd7dc7461a50172405a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7182d24cb9ed83c3e793bde3926531e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ad43aa4f62461977c2e13b01428b556c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4905618131160736, 0.4396417438983917, 0.38613632321357727, 0.30494746565818787], [0.48475587368011475, 0.07337385416030884, 0.278387188911438, 0.3098049461841583], [0.29222574830055237, 0.4004157781600952, 0.19416268169879913, 0.0951867327094078], [0.29222574830055237, 0.4004157781600952, 0.19416268169879913, 0.0951867327094078], [0.26826053857803345, 0.21500079333782196, 0.11182934045791626, 0.27236950397491455]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.3478876054286957, 0.4785013496875763, 0.05671229213476181, 0.23212681710720062], [0.25702235102653503, 0.06689565628767014, 0.20180051028728485, 0.24164846539497375], [0.3288941979408264, 0.4574975371360779, 0.30252569913864136, 0.07800009846687317], [0.3288941979408264, 0.4574975371360779, 0.30252569913864136, 0.07800009846687317], [0.40340694785118103, 0.1492466926574707, 0.29067420959472656, 0.32166600227355957]], dtype='float32').reshape([5, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5eedd941575a8025b1f6142513fa4c9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6323ab2df95c965ae5a18b5d2ee768b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66124f9c2801dcc6c590188b3dabe282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66124f9c2801dcc6c590188b3dabe282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66124f9c2801dcc6c590188b3dabe282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66124f9c2801dcc6c590188b3dabe282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66124f9c2801dcc6c590188b3dabe282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66124f9c2801dcc6c590188b3dabe282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66124f9c2801dcc6c590188b3dabe282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66124f9c2801dcc6c590188b3dabe282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66124f9c2801dcc6c590188b3dabe282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66124f9c2801dcc6c590188b3dabe282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66124f9c2801dcc6c590188b3dabe282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fadf83c28010ecd5bd8925902fc6620e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0906bc96c54199c50c23766f6e035159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6323ab2df95c965ae5a18b5d2ee768b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_266177bacdbc6e0a792e0f610c2e9b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15494532883167267, 0.2111409604549408, 0.16940054297447205, 0.07837673276662827], [0.22056446969509125, 0.49623557925224304, 0.30928391218185425, 0.13696753978729248], [0.1627940982580185, 0.29646024107933044, 0.267007052898407, 0.4014988839626312], [0.15494532883167267, 0.2111409604549408, 0.16940054297447205, 0.07837673276662827], [0.0711192935705185, 0.08745941519737244, 0.3294025659561157, 0.2732483148574829], [0.333394855260849, 0.2548178434371948, 0.49965474009513855, 0.017060736194252968], [0.0711192935705185, 0.08745941519737244, 0.3294025659561157, 0.2732483148574829]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.46240881085395813, 0.32869765162467957, 0.24423934519290924, 0.25821471214294434], [0.06261877715587616, 0.2492293268442154, 0.43208566308021545, 0.3304005563259125], [0.17018108069896698, 0.49496328830718994, 0.39770105481147766, 0.4668426513671875], [0.46240881085395813, 0.32869765162467957, 0.24423934519290924, 0.25821471214294434], [0.13235357403755188, 0.3676305413246155, 0.23703737556934357, 0.31373870372772217], [0.06706759333610535, 0.2564274072647095, 0.31626176834106445, 0.42102324962615967], [0.13235357403755188, 0.3676305413246155, 0.23703737556934357, 0.31373870372772217]], dtype='float32').reshape([7, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c1607c584dc7e104775c4f773344f40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_91a85761701982e53d3c9aefee45ccec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.04714285582304001]], [[0.24956832826137543]], [[0.06141399219632149]], [[0.27293410897254944]], [[0.020540714263916016]], [[0.47400450706481934]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6053178310394287]], [[0.6067324876785278]], [[0.7563060522079468]], [[0.6030699014663696]], [[0.6912596821784973]], [[0.6188269853591919]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_02ce98b787078fb5db949c316ce3dc84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.43553808331489563]], [[0.202225923538208]], [[0.1559573858976364]], [[0.39463528990745544]], [[0.45692548155784607]], [[0.21347668766975403]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6967204213142395]], [[0.5455881357192993]], [[0.774244487285614]], [[0.5381509065628052]], [[0.5265048742294312]], [[0.7456739544868469]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a929203e3767fdbf9f28631d4035808b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7837eb9c20d5f70502fabaaf04eced81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_75aeed24fc04a1265ee35faaeb56814b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_af7abc8f3c07a92290a9b6159b7d361b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dabba7fe0741275dfcdc1ed73b9802f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.009778277017176151, 0.02429402805864811]], [[0.0029977455269545317, 0.20873524248600006]], [[0.46495717763900757, 0.0797295793890953]], [[0.39304178953170776, 0.019942941144108772]], [[0.1783885955810547, 0.41458258032798767]], [[0.47772762179374695, 0.20179226994514465]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.358097106218338, 0.35106879472732544]], [[0.4313538074493408, 0.08216709643602371]], [[0.05587208271026611, 0.3657324016094208]], [[0.31227242946624756, 0.3250352442264557]], [[0.4644756019115448, 0.35592329502105713]], [[0.12418679893016815, 0.4918902516365051]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7eb8e2e75859be7d760cee2a82bf9707(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.3234972059726715, 0.20493970811367035]], [[0.3132672607898712, 0.17412687838077545]], [[0.014389144256711006, 0.3884662687778473]], [[0.3867090344429016, 0.06462222337722778]], [[0.17619918286800385, 0.07482553273439407]], [[0.027263877913355827, 0.33704501390457153]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.358097106218338, 0.35106879472732544]], [[0.4313538074493408, 0.08216709643602371]], [[0.05587208271026611, 0.3657324016094208]], [[0.31227242946624756, 0.3250352442264557]], [[0.4644756019115448, 0.35592329502105713]], [[0.12418679893016815, 0.4918902516365051]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_230bfc952c4fdced01186382fb750057(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.29133909940719604, 0.4853375554084778]], [[0.36761757731437683, 0.4571003019809723]], [[0.46794551610946655, 0.1688530147075653]], [[0.2670198976993561, 0.1982056200504303]], [[0.12026780098676682, 0.48663008213043213]], [[0.2773101329803467, 0.48897823691368103]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55e8ed499f2a3bcd8b91a2a8e96cbbb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
            paddle.to_tensor([0.3780723512172699, 0.10681550204753876, 0.21278166770935059, 0.23138809204101562, 0.32906362414360046, 0.22839735448360443, 0.13309429585933685, 0.0206376351416111, 0.05156761035323143, 0.46504005789756775, 0.13148869574069977, 0.449235200881958, 0.08905600011348724, 0.015455592423677444, 0.3863491714000702, 0.4747481942176819], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_25a7ceff1e2612cd942241b9353c5a2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3780723512172699, 0.10681550204753876, 0.21278166770935059, 0.23138809204101562, 0.32906362414360046, 0.22839735448360443, 0.13309429585933685, 0.0206376351416111, 0.05156761035323143, 0.46504005789756775, 0.13148869574069977, 0.449235200881958, 0.08905600011348724, 0.015455592423677444, 0.3863491714000702, 0.4747481942176819], dtype='float32').reshape([16]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_372aad51121867288b938e82c520cce6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_372aad51121867288b938e82c520cce6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6e02c6ca3972717e5b5b70e1ebc41e36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c58c6688c673867f3ad109f15937fb6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a95bfb21cf73e07e5e7d5e491f3804b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a95bfb21cf73e07e5e7d5e491f3804b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a95bfb21cf73e07e5e7d5e491f3804b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a95bfb21cf73e07e5e7d5e491f3804b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a95bfb21cf73e07e5e7d5e491f3804b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a95bfb21cf73e07e5e7d5e491f3804b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a95bfb21cf73e07e5e7d5e491f3804b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a95bfb21cf73e07e5e7d5e491f3804b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a95bfb21cf73e07e5e7d5e491f3804b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a95bfb21cf73e07e5e7d5e491f3804b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a95bfb21cf73e07e5e7d5e491f3804b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52d6c0904952e02947e994d334c88237(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38dc692e168c69c6d9272c552d7727e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c58c6688c673867f3ad109f15937fb6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9a650a7e9391931f30b34078c4bdac3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0660882219672203, 0.47173580527305603, 0.263930082321167, 0.1906377077102661], [0.08412470668554306, 0.41810542345046997, 0.11181697994470596, 0.21970145404338837], [0.15435844659805298, 0.23323611915111542, 0.2906814217567444, 0.22473807632923126], [0.43175461888313293, 0.4601755738258362, 0.09878799319267273, 0.04755397140979767], [0.2544768452644348, 0.316487193107605, 0.2239605039358139, 0.27877798676490784]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.4106596112251282, 0.14335723221302032, 0.02330786921083927, 0.11569497734308243], [0.42931506037712097, 0.01977924257516861, 0.230007603764534, 0.057321660220623016], [0.2798386216163635, 0.10481530427932739, 0.4301333725452423, 0.38091546297073364], [0.3798501193523407, 0.11627687513828278, 0.3571585416793823, 0.09328742325305939], [0.4864861071109772, 0.2632262408733368, 0.27916863560676575, 0.29254499077796936]], dtype='float32').reshape([5, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_753e4f9951039e8fb6243310705899f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1d596efcb9d2732b85dc08040cf23b5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2620193064212799, 0.11581896245479584, 0.053561560809612274, 0.08257486671209335], [0.03958377614617348, 0.3950630724430084, 0.20495222508907318, 0.3324486017227173], [0.3044680953025818, 0.4422242343425751, 0.3617860674858093, 0.3761495053768158], [0.03958377614617348, 0.3950630724430084, 0.20495222508907318, 0.3324486017227173], [0.3044680953025818, 0.4422242343425751, 0.3617860674858093, 0.3761495053768158]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.4025401175022125, 0.3901353180408478, 0.36967140436172485, 0.09798751771450043], [0.3619749844074249, 0.07483921945095062, 0.3436369299888611, 0.2519150972366333], [0.28524336218833923, 0.36912211775779724, 0.19368404150009155, 0.3241569995880127], [0.3619749844074249, 0.07483921945095062, 0.3436369299888611, 0.2519150972366333], [0.28524336218833923, 0.36912211775779724, 0.19368404150009155, 0.3241569995880127]], dtype='float32').reshape([5, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dbf0f22c4f3a7340ebbd00c64a90cf2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08568544685840607], [0.03848660737276077], [0.08805205672979355], [0.03215041384100914], [0.05588645488023758], [0.1058163195848465], [0.12910811603069305], [0.009429873898625374], [0.23962129652500153]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.39208778738975525], [0.3591558635234833], [0.15294159948825836], [0.27551141381263733], [0.2135521024465561], [0.3570266664028168], [0.3142800033092499], [0.4077095687389374], [0.48227444291114807]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1dd8422d0d5d1bc60990131be414319e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08652636408805847], [0.050644610077142715], [0.09682576358318329], [0.37938541173934937], [0.28843000531196594], [0.07827463001012802], [0.009334404021501541], [0.027887742966413498], [0.1417359560728073]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.49722179770469666], [0.4310191571712494], [0.3834795355796814], [0.30722612142562866], [0.2178652435541153], [0.28443190455436707], [0.2855451703071594], [0.22062554955482483], [0.498458594083786]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f3532d8815f2a1e2c847f8d2ee8e106(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36820653080940247], [0.03848660737276077], [0.08805205672979355], [0.03780742734670639], [0.23160281777381897], [0.18210230767726898], [0.21688763797283173], [0.009429873898625374], [0.23962129652500153]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.39208778738975525], [0.2094176560640335], [0.1283661276102066], [0.27551141381263733], [0.03479340672492981], [0.3570266664028168], [0.11812948435544968], [0.2456916868686676], [0.48227444291114807]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f9a778069b418f7d6f8803c18b3921f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39371946454048157], [0.050644610077142715], [0.09682576358318329], [0.37938541173934937], [0.37218934297561646], [0.31943804025650024], [0.009334404021501541], [0.027887742966413498], [0.1417359560728073]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.49722179770469666], [0.4310191571712494], [0.3834795355796814], [0.30722612142562866], [0.2178652435541153], [0.11710164695978165], [0.23001685738563538], [0.22062554955482483], [0.498458594083786]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4d62f21a69b8d75a4b3559a2e7458fd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08568544685840607], [0.1383717805147171], [0.2306109517812729], [0.03215041384100914], [0.05588645488023758], [0.1058163195848465], [0.12910811603069305], [0.02052994631230831], [0.24312002956867218]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.1547079086303711], [0.3591558635234833], [0.15294159948825836], [0.019754718989133835], [0.2135521024465561], [0.2554653286933899], [0.3142800033092499], [0.4077095687389374], [0.42899003624916077]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dfcef0fef8fab4a05f88bbd205b63c07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08652636408805847], [0.38375985622406006], [0.22444428503513336], [0.3983890116214752], [0.28843000531196594], [0.07827463001012802], [0.3716101050376892], [0.20676524937152863], [0.260421484708786]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.39656439423561096], [0.12202204018831253], [0.0054341829381883144], [0.014495838433504105], [0.0013494952581822872], [0.28443190455436707], [0.2855451703071594], [0.20876038074493408], [0.38022154569625854]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6ab8d4e15efa4ab6685b1d4b16a9055c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.023871352896094322], [0.0072302743792533875], [0.028566552326083183], [-0.012393927201628685], [-0.014890296384692192], [-0.004542335867881775], [-0.037730999290943146], [0.046309053897857666], [0.10882711410522461]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_57300b75399eb3a297d1b5ae346d5b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36820653080940247], [0.1383717805147171], [0.2306109517812729], [0.03780742734670639], [0.23160281777381897], [0.18210230767726898], [0.21688763797283173], [0.02052994631230831], [0.24312002956867218]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.1547079086303711], [0.2094176560640335], [0.1283661276102066], [0.019754718989133835], [0.03479340672492981], [0.2554653286933899], [0.11812948435544968], [0.2456916868686676], [0.42899003624916077]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e99dda31b438944e774c2dd8272ee17a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39371946454048157], [0.38375985622406006], [0.22444428503513336], [0.3983890116214752], [0.37218934297561646], [0.31943804025650024], [0.3716101050376892], [0.20676524937152863], [0.260421484708786]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.39656439423561096], [0.12202204018831253], [0.0054341829381883144], [0.014495838433504105], [0.0013494952581822872], [0.11710164695978165], [0.23001685738563538], [0.20876038074493408], [0.38022154569625854]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_473ed62fcf0ad6ec79b1600d60c809ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.000607388443313539], [-0.018595393747091293], [0.022392649203538895], [0.00693031121045351], [0.07298476994037628], [-0.014844009652733803], [0.013983488082885742], [0.00044922734377905726], [0.022267237305641174]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.023871352896094322], [0.0072302743792533875], [0.028566552326083183], [-0.012393927201628685], [-0.014890296384692192], [-0.004542335867881775], [-0.037730999290943146], [0.046309053897857666], [0.10882711410522461]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_abcce45492f97a3cf126b8f399949a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [-0.0], [-0.0], [-0.0], [-0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[40.3016242980957], [1.388820767402649], [-0.27571114897727966], [2.788365364074707], [1.20401930809021], [0.6939953565597534], [3.698253870010376], [-102.08600616455078], [-3.887320041656494]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b2649627c56cc22f68d3cbea932371b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_acb2b446c113941f8ae47e1572ee95c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.06146731972694397]], [[0.38457757234573364]], [[0.1499379724264145]], [[0.2519291639328003]], [[0.4287847876548767]], [[0.48430103063583374]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.7040383219718933]], [[0.5041806101799011]], [[0.5248106718063354]], [[0.5276604890823364]], [[0.8099044561386108]], [[0.5424766540527344]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e307239dd03430cb7079aef3b0aa6f2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.03920156881213188]], [[0.022001300007104874]], [[0.24809589982032776]], [[0.39565593004226685]], [[0.10346271097660065]], [[0.42565131187438965]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.7184906005859375]], [[0.5257463455200195]], [[0.6536774635314941]], [[0.6953741312026978]], [[0.7265492677688599]], [[0.5624583959579468]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f4efdd4f4553731eda82c2570b8cda74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_792aae5d0779f8e92a53f142d5ffab77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d9c0599eb4cf3f8061d85583c865735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d9c0599eb4cf3f8061d85583c865735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d9c0599eb4cf3f8061d85583c865735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d9c0599eb4cf3f8061d85583c865735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d9c0599eb4cf3f8061d85583c865735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d9c0599eb4cf3f8061d85583c865735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d9c0599eb4cf3f8061d85583c865735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d9c0599eb4cf3f8061d85583c865735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d9c0599eb4cf3f8061d85583c865735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d9c0599eb4cf3f8061d85583c865735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d9c0599eb4cf3f8061d85583c865735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03b1906bab8cf1f48b4606ade5191acd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f0222eca32e6403f4049a2fb3b688ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_792aae5d0779f8e92a53f142d5ffab77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2ee5247c11c6647ab647c3ba412596ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13856299221515656, 0.4353436827659607, 0.2094043344259262, 0.2879944443702698], [0.08696585893630981, 0.2929264307022095, 0.17700904607772827, 0.2517296075820923], [0.18547385931015015, 0.4464535713195801, 0.4669950008392334, 0.4609496593475342], [0.08696585893630981, 0.2929264307022095, 0.17700904607772827, 0.2517296075820923], [0.18547385931015015, 0.4464535713195801, 0.4669950008392334, 0.4609496593475342], [0.04176447167992592, 0.43585583567619324, 0.007390841841697693, 0.41927239298820496], [0.04176447167992592, 0.43585583567619324, 0.007390841841697693, 0.41927239298820496]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.3110031485557556, 0.12883557379245758, 0.014142144471406937, 0.29536378383636475], [0.03524753451347351, 0.44889235496520996, 0.32588690519332886, 0.4541289508342743], [0.3971027433872223, 0.4102397561073303, 0.28529155254364014, 0.39603105187416077], [0.03524753451347351, 0.44889235496520996, 0.32588690519332886, 0.4541289508342743], [0.3971027433872223, 0.4102397561073303, 0.28529155254364014, 0.39603105187416077], [0.43009355664253235, 0.49960651993751526, 0.3261641263961792, 0.4162203073501587], [0.43009355664253235, 0.49960651993751526, 0.3261641263961792, 0.4162203073501587]], dtype='float32').reshape([7, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e93be1b02e7657a17ee401eb2ea5cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e93be1b02e7657a17ee401eb2ea5cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6a0d88c3b31ac0ec78be6d8038ebdb82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_18b385f2773e20549fa68a071089e091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8b8646482c76ef0cc3a8840014a75d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2608264982700348, 0.4725295603275299, 0.30785173177719116, 0.08038189262151718, 0.0575774721801281, 0.3921468257904053], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1027945876121521, 0.3881237208843231, 0.152043879032135, 0.21335071325302124, 0.38008421659469604, 0.12074317038059235], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ae96253cfd7568e986d861da85f3af2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4957810342311859, 0.08107513189315796, 0.33436471223831177, 0.4922436773777008, 0.36378422379493713, 0.13394984602928162], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2527327835559845, 0.1542942374944687, 0.470980703830719, 0.32545068860054016, 0.27355799078941345, 0.35376250743865967], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_54ff2d0396fef3ecaa5346e0d5811295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24978351593017578, 0.11446373909711838, 0.06451474130153656, 0.04623079672455788, 0.3613818883895874, 0.4987070560455322], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2304920107126236, 0.0650036409497261, 0.457919716835022, 0.030212104320526123, 0.35011738538742065, 0.12877273559570312], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8fc350a0642ca1eee42599eb3bcda7e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2845926582813263, 0.16627027094364166, 0.25496557354927063, 0.29948753118515015, 0.21094956994056702, 0.2771330177783966], dtype='float32').reshape([6]),
            paddle.to_tensor([0.23068910837173462, 0.1622818112373352, 0.3164260983467102, 0.16976673901081085, 0.15035927295684814, 0.22009573876857758], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_989f2114979a7884f0b0ca404ba695ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24978351593017578, 0.11446373909711838, 0.06451474130153656, 0.04623079672455788, 0.3613818883895874, 0.3921468257904053], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2304920107126236, 0.3881237208843231, 0.457919716835022, 0.21335071325302124, 0.38008421659469604, 0.12877273559570312], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8cff451749268b28a898dd3399de8eb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2845926582813263, 0.1542942374944687, 0.25496557354927063, 0.29948753118515015, 0.21094956994056702, 0.2771330177783966], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2527327835559845, 0.1622818112373352, 0.470980703830719, 0.32545068860054016, 0.27355799078941345, 0.35376250743865967], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c4cefd8604ec773bae6ee8cd8740d8db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2608264982700348, 0.4725295603275299, 0.30785173177719116, 0.21335071325302124, 0.38008421659469604, 0.3921468257904053], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1027945876121521, 0.3881237208843231, 0.152043879032135, 0.21335071325302124, 0.38008421659469604, 0.12074317038059235], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ade7d3f0ac72a7e36b62e89bbd6b2cc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4957810342311859, 0.1542942374944687, 0.470980703830719, 0.4922436773777008, 0.36378422379493713, 0.35376250743865967], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2527327835559845, 0.1542942374944687, 0.470980703830719, 0.32545068860054016, 0.27355799078941345, 0.35376250743865967], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b31f71a1dc3b2bd3a84e6516c438c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.039449259638786316, 0.00019726960454136133, 0.024178875610232353, 0.002077957382425666, 0.0006825195741839707, 0.02110004797577858], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0006146249361336231, 0.0, 0.0, 0.0, 0.0, -0.0], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52d3eee2ce8e716ae1212a9dea1b69a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18181054294109344, 0.4303266406059265, 0.22994780540466309, 0.1468663066625595, 0.21883083879947662, 0.2564449906349182], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2401377558708191, 0.08973369002342224, 0.26121723651885986, 0.03822144865989685, 0.35574963688850403, 0.3137398958206177], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_239165a78e3bbb76d225e4f3a0cd1477(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3742569088935852, 0.11768468469381332, 0.4026727080345154, 0.4088471829891205, 0.3186711072921753, 0.24385617673397064], dtype='float32').reshape([6]),
            paddle.to_tensor([0.25764089822769165, 0.16427603363990784, 0.2856958508491516, 0.2346271276473999, 0.18065442144870758, 0.2486143708229065], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_64c1ad6254405a9fadac3cd392be9c4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2608264982700348, 0.4725295603275299, 0.30785173177719116, 0.21335071325302124, 0.38008421659469604, 0.4987070560455322], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1027945876121521, 0.0650036409497261, 0.152043879032135, 0.030212104320526123, 0.35011738538742065, 0.12074317038059235], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9ce8b4d942fb0b384c967fa1d9e3c3f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4957810342311859, 0.16627027094364166, 0.470980703830719, 0.4922436773777008, 0.36378422379493713, 0.35376250743865967], dtype='float32').reshape([6]),
            paddle.to_tensor([0.23068910837173462, 0.1542942374944687, 0.3164260983467102, 0.16976673901081085, 0.15035927295684814, 0.22009573876857758], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c1be02e8773f94c2bee9eb11fee73ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3436858057975769, 1.4903305768966675, 1.41582190990448, 0.12286394089460373, 0.18381407856941223, 1.4178187847137451], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5765213966369629, -0.856249988079071, -0.8509345054626465, -0.6730365753173828, -1.298005223274231, -0.8900416493415833], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a72878964a0e483b4d3c4102d94541d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf6534e22c43df77b7a5aea125cba6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf6534e22c43df77b7a5aea125cba6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf6534e22c43df77b7a5aea125cba6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf6534e22c43df77b7a5aea125cba6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf6534e22c43df77b7a5aea125cba6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf6534e22c43df77b7a5aea125cba6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf6534e22c43df77b7a5aea125cba6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf6534e22c43df77b7a5aea125cba6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf6534e22c43df77b7a5aea125cba6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf6534e22c43df77b7a5aea125cba6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf6534e22c43df77b7a5aea125cba6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52d6c0904952e02947e994d334c88237(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38dc692e168c69c6d9272c552d7727e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a72878964a0e483b4d3c4102d94541d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e426acc2a8a3c866ade65ea71932622b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_69cefdcc79ddad326fae6984d3c01b5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([0.16727307438850403, 0.29837217926979065, 0.19894284009933472, 0.48071572184562683, 0.3676881194114685, 0.4198280870914459, 0.17023754119873047, 0.16682079434394836, 0.41211000084877014, 0.27952146530151367, 0.03478027507662773, 0.25146594643592834, 0.3347516655921936, 0.07699854671955109, 0.16147416830062866, 0.18425174057483673, 0.1390560418367386, 0.17906422913074493, 0.1935458481311798, 0.2319454848766327, 0.13256007432937622, 0.057637352496385574, 0.14197881519794464, 0.27830740809440613], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8d5c4ef7476afda985a09d0fd3bd5879(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.16727307438850403, 0.29837217926979065, 0.19894284009933472, 0.48071572184562683, 0.3676881194114685, 0.4198280870914459, 0.17023754119873047, 0.16682079434394836, 0.41211000084877014, 0.27952146530151367, 0.03478027507662773, 0.25146594643592834, 0.3347516655921936, 0.07699854671955109, 0.16147416830062866, 0.18425174057483673, 0.1390560418367386, 0.17906422913074493, 0.1935458481311798, 0.2319454848766327, 0.13256007432937622, 0.057637352496385574, 0.14197881519794464, 0.27830740809440613], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b747f17c9767b2f817cd1e5c98fbdf75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_919b78e5f2f145edf32c93c4595e1139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_919b78e5f2f145edf32c93c4595e1139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_919b78e5f2f145edf32c93c4595e1139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_919b78e5f2f145edf32c93c4595e1139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_919b78e5f2f145edf32c93c4595e1139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_919b78e5f2f145edf32c93c4595e1139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_919b78e5f2f145edf32c93c4595e1139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_919b78e5f2f145edf32c93c4595e1139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_919b78e5f2f145edf32c93c4595e1139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_919b78e5f2f145edf32c93c4595e1139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_919b78e5f2f145edf32c93c4595e1139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3ed75efcd84ec090d0b5f8a4bdcffa95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1d0e3e7e00154ab42eff6103c79fdb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b747f17c9767b2f817cd1e5c98fbdf75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a4e823eb61d9b61aba09226acd691f40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
            paddle.to_tensor([0.09576865285634995, 0.21895988285541534, 0.47236356139183044, 0.28012216091156006], dtype='float32').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bb686935ff7497e52ed3e1c147aa8a15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09576865285634995, 0.21895988285541534, 0.47236356139183044, 0.28012216091156006], dtype='float32').reshape([4]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([4]),
        ]



class PrimitiveOp_4d28e76b50ed241aed2c5946f80248f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_88abd78c6b8823b63b747f20ccf526af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d28e76b50ed241aed2c5946f80248f2
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int32').reshape([1]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6de5048b4b9678ffe6cdcce96f01d92f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d28e76b50ed241aed2c5946f80248f2
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d9a63d5e912ab50e13dd8f29dcfd2c40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.48869359493255615, 0.25505194067955017, 0.2965860962867737, 0.1435116082429886], [0.29652026295661926, 0.17479072511196136, 0.12994545698165894, 0.3918682336807251], [0.03457598015666008, 0.21453428268432617, 0.40904825925827026, 0.13430796563625336], [0.4367668628692627, 0.10261180251836777, 0.21258141100406647, 0.15890343487262726], [0.4367668628692627, 0.10261180251836777, 0.21258141100406647, 0.15890343487262726], [0.03457598015666008, 0.21453428268432617, 0.40904825925827026, 0.13430796563625336]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.17786620557308197, 0.3837328553199768, 0.4018065929412842, 0.052461110055446625], [0.1788792461156845, 0.17232021689414978, 0.44501304626464844, 0.18641409277915955], [0.23814435303211212, 0.0061455052345991135, 0.03816644474864006, 0.24216414988040924], [0.18565472960472107, 0.4981105327606201, 0.3184441328048706, 0.44102829694747925], [0.18565472960472107, 0.4981105327606201, 0.3184441328048706, 0.44102829694747925], [0.23814435303211212, 0.0061455052345991135, 0.03816644474864006, 0.24216414988040924]], dtype='float32').reshape([6, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a5ae8a733e11dc71ae541a385555c8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.31454479694366455, 0.10439124703407288, 0.16489987075328827, 0.18370254337787628], [0.11546322703361511, 0.26400521397590637, 0.06056874617934227, 0.09392143040895462], [0.4271167516708374, 0.453997939825058, 0.2668359875679016, 0.30915793776512146], [0.05479086562991142, 0.4311014413833618, 0.2002004086971283, 0.21852238476276398], [0.31454479694366455, 0.10439124703407288, 0.16489987075328827, 0.18370254337787628]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.26386094093322754, 0.27112436294555664, 0.39497309923171997, 0.42580345273017883], [0.47162744402885437, 0.0874064713716507, 0.2927992045879364, 0.4740701913833618], [0.15716779232025146, 0.0017121427226811647, 0.13402105867862701, 0.2784697711467743], [0.2673611044883728, 0.20431914925575256, 0.4392146170139313, 0.18085090816020966], [0.26386094093322754, 0.27112436294555664, 0.39497309923171997, 0.42580345273017883]], dtype='float32').reshape([5, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_56bd3d7f1023b32564a39a9793b4d622(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1cbcd7991ed6fdb0c9c5b0ce675977d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11903903633356094]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3555246889591217]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d41f96ee1b152bb09b6faa5f12d868b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0831770971417427]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.45340871810913086]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05f1cc7c1bf89a05d469eb24c2730477(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33794644474983215]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2977246940135956]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ad03b962229cddadf6f2e39118379806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12983585894107819]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.45340871810913086]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1cbcd7991ed6fdb0c9c5b0ce675977d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11903903633356094]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3555246889591217]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a06e94cca90c7caf756fbb7ab054eaab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0831770971417427]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.43772584199905396]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9553c98238d3d876f3f0e0fe7f49f6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07083103060722351]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05f1cc7c1bf89a05d469eb24c2730477(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33794644474983215]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2977246940135956]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a0b8984f934abc666cf9a570906a85b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12983585894107819]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.43772584199905396]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6e566f0e9601517bfacd01266c4611de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.012383874505758286]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.07083103060722351]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_594049bfdd18338219bc9314c2f732df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[6.719618320465088]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5238a0ed6b632fd19c88e5f35d3f4b47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20998290181159973], [0.18442489206790924], [0.1968512237071991], [0.20288529992103577], [0.21599259972572327], [0.19651634991168976]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3895230293273926], [0.4666270613670349], [0.17671829462051392], [0.3581591248512268], [0.42263373732566833], [0.24344341456890106]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_792a83a8650844cd06cca95700a8fbf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4151418209075928], [0.2509276866912842], [0.13929075002670288], [0.03818127512931824], [0.08567338436841965], [0.23274247348308563]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.4103717803955078], [0.39778658747673035], [0.40150901675224304], [0.42183321714401245], [0.1773831695318222], [0.48681026697158813]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_789850f723c6754bce6385edefe0a3f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20998290181159973], [0.4498758018016815], [0.24683597683906555], [0.20288529992103577], [0.21599259972572327], [0.19651634991168976]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.15491808950901031], [0.16919776797294617], [0.1292070597410202], [0.12714418768882751], [0.33482757210731506], [0.2098187357187271]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dc789da2137c74034b4363355f450b61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4901370406150818], [0.2509276866912842], [0.2429288774728775], [0.06132770702242851], [0.08567338436841965], [0.23274247348308563]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.4103717803955078], [0.39778658747673035], [0.40150901675224304], [0.42183321714401245], [0.09941722452640533], [0.25498807430267334]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_306651a096272fa6d733b41ea8025470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22682112455368042], [0.18442489206790924], [0.1968512237071991], [0.46012169122695923], [0.40794169902801514], [0.4862612187862396]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3895230293273926], [0.4666270613670349], [0.17671829462051392], [0.3581591248512268], [0.42263373732566833], [0.24344341456890106]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c77c433ea911e2411aac58d395d8556e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4151418209075928], [0.3606884777545929], [0.13929075002670288], [0.03818127512931824], [0.28175589442253113], [0.3325986862182617]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.1010279506444931], [0.2909882366657257], [0.2298392504453659], [0.29276591539382935], [0.1773831695318222], [0.48681026697158813]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1606ef2561d5873d54f91490159d98b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0467146635055542], [-0.060889631509780884], [-0.02047661691904068], [-0.05326319485902786], [9.980075992643833e-05], [-0.037149399518966675]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6f8b324f7e7b0868e1c9528cbc26d0cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22682112455368042], [0.4498758018016815], [0.24683597683906555], [0.46012169122695923], [0.40794169902801514], [0.4862612187862396]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.15491808950901031], [0.16919776797294617], [0.1292070597410202], [0.12714418768882751], [0.33482757210731506], [0.2098187357187271]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ecb901fe7978d0cea63d57c249161309(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4901370406150818], [0.3606884777545929], [0.2429288774728775], [0.06132770702242851], [0.28175589442253113], [0.3325986862182617]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.1010279506444931], [0.2909882366657257], [0.2298392504453659], [0.29276591539382935], [0.09941722452640533], [0.25498807430267334]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_264093c99d91c0e58506e35590b0d9ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.02797812409698963], [0.019563326612114906], [0.0015397188253700733], [-0.07706371694803238], [0.01333153247833252], [0.021454868838191032]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.0467146635055542], [-0.060889631509780884], [-0.02047661691904068], [-0.05326319485902786], [9.980086178984493e-05], [-0.037149399518966675]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0a42593a2c026c8062f5eede8020720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [-0.0], [-0.0], [0.0], [-0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[2.6696853637695312], [4.1124372482299805], [14.298932075500488], [0.30884212255477905], [0.9925139546394348], [2.731513738632202]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc98259315ae6586ac9df6a85b6c5562(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1824364960193634, 0.11383223533630371, 0.15641945600509644, 0.39408645033836365], [0.44869938492774963, 0.021236063912510872, 0.01257316768169403, 0.1094929650425911], [0.23305965960025787, 0.25187548995018005, 0.1086021214723587, 0.40324482321739197], [0.4280799329280853, 0.38631755113601685, 0.44555965065956116, 0.33409902453422546]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([[0.1865062266588211, 0.10327757894992828, 0.31373849511146545, 0.37114739418029785], [0.4478083550930023, 0.3082262873649597, 0.22778934240341187, 0.13071022927761078], [0.07917411625385284, 0.3322516679763794, 0.15485522150993347, 0.46097493171691895], [0.12284959852695465, 0.0587301105260849, 0.26185959577560425, 0.17734073102474213]], dtype='float32').reshape([4, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ad01ba209aa563c77d8a0f5ecf6c7ca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_91529f4ce60f8760c837e733a685a65d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc789a49551a286bc9469df13afb9726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc789a49551a286bc9469df13afb9726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc789a49551a286bc9469df13afb9726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc789a49551a286bc9469df13afb9726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc789a49551a286bc9469df13afb9726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc789a49551a286bc9469df13afb9726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc789a49551a286bc9469df13afb9726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc789a49551a286bc9469df13afb9726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc789a49551a286bc9469df13afb9726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc789a49551a286bc9469df13afb9726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc789a49551a286bc9469df13afb9726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15aa17eea687877cf988c12a2c059c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_358bc8c1eb98b6e964028dcc0f56dec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_91529f4ce60f8760c837e733a685a65d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a9ea8ff0bae1a32aa2249436f11ffaaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.21805590391159058, 0.369320809841156, 0.46295464038848877, 0.32399559020996094], [0.21805590391159058, 0.369320809841156, 0.46295464038848877, 0.32399559020996094], [0.45063576102256775, 0.08473407477140427, 0.2964290678501129, 0.17995186150074005], [0.35198330879211426, 0.0768512561917305, 0.0005001812824048102, 0.43629372119903564], [0.23088055849075317, 0.3435775935649872, 0.022802097722887993, 0.006141966208815575], [0.1841537058353424, 0.24625396728515625, 0.19545932114124298, 0.41149935126304626], [0.0622473768889904, 0.010412028059363365, 0.32384076714515686, 0.019491543993353844]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.11718153953552246, 0.18766602873802185, 0.3853262960910797, 0.17273500561714172], [0.11718153953552246, 0.18766602873802185, 0.3853262960910797, 0.17273500561714172], [0.11498585343360901, 0.39520618319511414, 0.019509049132466316, 0.21388070285320282], [0.11598620563745499, 0.13958659768104553, 0.2296896129846573, 0.40918001532554626], [0.08412293344736099, 0.36157235503196716, 0.21181900799274445, 0.0409696027636528], [0.4178256392478943, 0.11422522366046906, 0.3132604658603668, 0.10981956124305725], [0.3702118396759033, 0.0816604271531105, 0.2898832857608795, 0.05852610245347023]], dtype='float32').reshape([7, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40f637a1c203bb54d195c3251937c47a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_691fd97a65d1ec77739972ae555392e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a8ac04f93f4fc769ff3ade2683d2b9e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cacf6ebbc2a34067425e5653b7587486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cacf6ebbc2a34067425e5653b7587486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cacf6ebbc2a34067425e5653b7587486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cacf6ebbc2a34067425e5653b7587486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cacf6ebbc2a34067425e5653b7587486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cacf6ebbc2a34067425e5653b7587486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cacf6ebbc2a34067425e5653b7587486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cacf6ebbc2a34067425e5653b7587486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cacf6ebbc2a34067425e5653b7587486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cacf6ebbc2a34067425e5653b7587486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cacf6ebbc2a34067425e5653b7587486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fcb8107f4be5eea7babae7f8cd89696e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f56d38e40cb258115066d09088cf23a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a8ac04f93f4fc769ff3ade2683d2b9e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a11065e362014a6c69078a814da23d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0529d819cea1398b9a6635ac513e131f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0529d819cea1398b9a6635ac513e131f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0529d819cea1398b9a6635ac513e131f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0529d819cea1398b9a6635ac513e131f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0529d819cea1398b9a6635ac513e131f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0529d819cea1398b9a6635ac513e131f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0529d819cea1398b9a6635ac513e131f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0529d819cea1398b9a6635ac513e131f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0529d819cea1398b9a6635ac513e131f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0529d819cea1398b9a6635ac513e131f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0529d819cea1398b9a6635ac513e131f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_21a50b949a46359c33f1a60837b3d774(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67dd1ecbca2884ee78c7cd1a7eea2215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a11065e362014a6c69078a814da23d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_515f52247eb0925eb57e78c260d58f96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3a19ffbad53f86cf6124faf17bca69f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1983117163181305, 0.4262564480304718, 0.44021251797676086, 0.45081812143325806], [0.3675566613674164, 0.2470126450061798, 0.4268859326839447, 0.33921435475349426], [0.3675566613674164, 0.2470126450061798, 0.4268859326839447, 0.33921435475349426], [0.33047568798065186, 0.14772096276283264, 0.16170357167720795, 0.1762121468782425], [0.1535663604736328, 0.11277178674936295, 0.3490486741065979, 0.19976338744163513], [0.052271824330091476, 0.46588462591171265, 0.35674288868904114, 0.013300606049597263]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.4848327338695526, 0.167837992310524, 0.48880672454833984, 0.4181344211101532], [0.32352542877197266, 0.27273812890052795, 0.30801448225975037, 0.2090080976486206], [0.32352542877197266, 0.27273812890052795, 0.30801448225975037, 0.2090080976486206], [0.005183657631278038, 0.49299538135528564, 0.12437743693590164, 0.14061975479125977], [0.10720117390155792, 0.40686947107315063, 0.33792147040367126, 0.4700213372707367], [0.44660133123397827, 0.26956701278686523, 0.28133127093315125, 0.14439621567726135]], dtype='float32').reshape([6, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_293d93bbcf6e674049780843ff641a48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_97dfafb64bccf9246bec94eb095d109b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.39817842841148376, 12.484559059143066, 2.655482292175293, 1.6678518056869507], [2.003458261489868, 1.472330927848816, 0.9269778728485107, 0.15994922816753387]], dtype='float32').reshape([2, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07ed5983b01422c9e20bc8c43f0c8f5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5506a7d5add76bd5c93be0c3ec95da8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89849624a078fdf071708d8f96e5f22a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1.776720643043518, 0.14957483112812042, 0.3515414893627167, 1.572020173072815], [4.945065021514893, 5.548974990844727, 1.1182193756103516, 0.07188649475574493]], dtype='float32').reshape([2, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6e064d47fc2ef1aad541cfaac0081039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27102693915367126], [0.21676185727119446], [0.3153221905231476], [0.01315593346953392], [0.1262916475534439]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4706946909427643], [0.39319300651550293], [0.19910383224487305], [0.3212777376174927], [0.29648688435554504]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cac090a0c6006c4ff68fabaed33f11a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39167019724845886], [0.046127624809741974], [0.04106692224740982], [0.014858669601380825], [0.17310704290866852]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.16876435279846191], [0.1894516497850418], [0.3004133403301239], [0.4173230230808258], [0.3782731890678406]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68f0d019e4c9fd642d19ba5994ae6632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3077138066291809], [0.21676185727119446], [0.36836960911750793], [0.01315593346953392], [0.1262916475534439]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.3776218891143799], [0.017230553552508354], [0.12865380942821503], [0.0789160430431366], [0.29648688435554504]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46a3ac494b39c23c2c6c2ad758ea68ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39167019724845886], [0.046127624809741974], [0.13554736971855164], [0.2480829954147339], [0.17310704290866852]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.04191706329584122], [0.1894516497850418], [0.11870089173316956], [0.0004433089052326977], [0.28320610523223877]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77363bca5aab326a7078b852be665c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27102693915367126], [0.3185052275657654], [0.3153221905231476], [0.430990993976593], [0.4189223051071167]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4706946909427643], [0.39319300651550293], [0.19910383224487305], [0.3212777376174927], [0.27562597393989563]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_16a674fb86173a0a7b96accf14c24b02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4086301028728485], [0.391907662153244], [0.04106692224740982], [0.014858669601380825], [0.4931931793689728]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.16876435279846191], [0.1280040442943573], [0.3004133403301239], [0.4173230230808258], [0.3782731890678406]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_26701ed506c04a30ea9dbe81ee2556d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.07234402745962143], [-0.048308003693819046], [-0.026102447882294655], [-0.060440488159656525], [0.03520594909787178]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3c14aeb1c83e16814c4b64af6530f2a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3077138066291809], [0.3185052275657654], [0.36836960911750793], [0.430990993976593], [0.4189223051071167]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.3776218891143799], [0.017230553552508354], [0.12865380942821503], [0.0789160430431366], [0.27562597393989563]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15432243cbc4d9fe3899ec0e1153673d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4086301028728485], [0.391907662153244], [0.13554736971855164], [0.2480829954147339], [0.4931931793689728]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.04191706329584122], [0.1280040442943573], [0.11870089173316956], [0.0004433089052326977], [0.28320610523223877]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32464289f44b3279d82a52003896bdb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.025636205449700356], [0.07950747758150101], [0.0040383669547736645], [0.08718772977590561], [0.03009037673473358]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.07234402745962143], [-0.048308003693819046], [-0.026102447882294655], [-0.060440488159656525], [0.03520594909787178]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6aaa08ff2c87645dd7e343b5022045c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-1.821947693824768], [1.6075907945632935], [7.4636149406433105], [1.6932224035263062], [-0.1700069159269333]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9f31b45242dcc5467d9b1b85f0c9a8f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ecb9439dd0e465697068e88f303ba79e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77134f31c324c9851cefae55031bdacc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77134f31c324c9851cefae55031bdacc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77134f31c324c9851cefae55031bdacc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77134f31c324c9851cefae55031bdacc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77134f31c324c9851cefae55031bdacc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77134f31c324c9851cefae55031bdacc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77134f31c324c9851cefae55031bdacc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77134f31c324c9851cefae55031bdacc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77134f31c324c9851cefae55031bdacc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77134f31c324c9851cefae55031bdacc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77134f31c324c9851cefae55031bdacc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0cf1b79d6b0bda99fe30897e159b4ad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_482a57e3f8a9d1754788fd87985c8f0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ecb9439dd0e465697068e88f303ba79e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_db5d863d15f5409acda9efda25784065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1af9f4e8b51e4a32f271137bb261886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1af9f4e8b51e4a32f271137bb261886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1af9f4e8b51e4a32f271137bb261886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1af9f4e8b51e4a32f271137bb261886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1af9f4e8b51e4a32f271137bb261886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1af9f4e8b51e4a32f271137bb261886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1af9f4e8b51e4a32f271137bb261886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1af9f4e8b51e4a32f271137bb261886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1af9f4e8b51e4a32f271137bb261886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1af9f4e8b51e4a32f271137bb261886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1af9f4e8b51e4a32f271137bb261886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8d5f52ddf8383c305d82d62c38a7f453(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc25a83b16b04b6ad5a3de78a8e76b5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_db5d863d15f5409acda9efda25784065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_73c0d011019499def6ab3f217d4bd8f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_152f8cd835eb75edcae9c8969dda1351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_152f8cd835eb75edcae9c8969dda1351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_152f8cd835eb75edcae9c8969dda1351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_152f8cd835eb75edcae9c8969dda1351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_152f8cd835eb75edcae9c8969dda1351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_152f8cd835eb75edcae9c8969dda1351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_152f8cd835eb75edcae9c8969dda1351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_152f8cd835eb75edcae9c8969dda1351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_152f8cd835eb75edcae9c8969dda1351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_152f8cd835eb75edcae9c8969dda1351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_152f8cd835eb75edcae9c8969dda1351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dba09ba03ffefda8f2726c71385338fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9f8e1c85db4557fff30430af9b30c7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_73c0d011019499def6ab3f217d4bd8f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80c271ec998c36d91accf2075aa68a64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3a508b496778ed0b643bc983d4abc57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e76620bcf0d595eda1ab5ed69cc6b979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4bd2d697bf6182f52895306cc885df7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([0.4473949074745178, 0.08388441801071167, 0.17490623891353607, 0.2467467337846756, 0.14741015434265137, 0.14820529520511627, 0.4406585693359375, 0.37450501322746277, 0.04052567854523659, 0.48126161098480225, 0.4376707375049591, 0.2969852685928345, 0.2029794454574585, 0.15935193002223969, 0.2048863172531128, 0.25172749161720276, 0.38391077518463135, 0.35427677631378174, 0.34242749214172363, 0.49203726649284363], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_09380d0d6e063af8224ede249c2f0ebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4473949074745178, 0.08388441801071167, 0.17490623891353607, 0.2467467337846756, 0.14741015434265137, 0.14820529520511627, 0.4406585693359375, 0.37450501322746277, 0.04052567854523659, 0.48126161098480225, 0.4376707375049591, 0.2969852685928345, 0.2029794454574585, 0.15935193002223969, 0.2048863172531128, 0.25172749161720276, 0.38391077518463135, 0.35427677631378174, 0.34242749214172363, 0.49203726649284363], dtype='float32').reshape([20]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_01bb10db17854f1b70da84dea0683b7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1317010074853897], [0.1832035928964615], [0.005624867510050535], [0.382903516292572]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4710065722465515], [0.27305787801742554], [0.37650251388549805], [0.3584076464176178]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1a0b2eb921f5a565e0e6ec6b8f868f8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2849007844924927], [0.1524905115365982], [0.36212438344955444], [0.4171924889087677]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3558596074581146], [0.34534937143325806], [0.45495909452438354], [0.43209323287010193]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b057d51f5ce2a7ef4b1604848a7e43f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1317010074853897], [0.1832035928964615], [0.005624867510050535], [0.382903516292572]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4710065722465515], [0.27305787801742554], [0.1443967968225479], [0.18359437584877014]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f54ee33ad5572909cc204d2f34c47ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.407511442899704], [0.41122034192085266], [0.36212438344955444], [0.47452500462532043]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3558596074581146], [0.24631327390670776], [0.38240188360214233], [0.43209323287010193]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8757f20916feccccae702f6990bca23f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.21841885149478912], [0.4341205954551697], [0.17934243381023407], [0.48898959159851074]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.11861217021942139], [0.1668710708618164], [0.37650251388549805], [0.3584076464176178]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dc8042352da3be90258765fe14bec82f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2849007844924927], [0.1524905115365982], [0.386353075504303], [0.4171924889087677]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.17836087942123413], [0.34534937143325806], [0.45495909452438354], [0.29443272948265076]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62106d4632ce0001929f0696891a2a61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.006892362609505653], [-0.06635904312133789], [0.016340315341949463], [0.024487247690558434]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47d77aaf788cd9fc4f442a835c572b03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.21841885149478912], [0.4341205954551697], [0.17934243381023407], [0.48898959159851074]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.11861217021942139], [0.1668710708618164], [0.1443967968225479], [0.18359437584877014]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f9ba8564672e52eccfb9552453f64491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.407511442899704], [0.41122034192085266], [0.386353075504303], [0.47452500462532043]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.17836087942123413], [0.24631327390670776], [0.38240188360214233], [0.29443272948265076]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb8ae06b21b230a17f188b9522134849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.022870756685733795], [0.04407133534550667], [0.00013807702634949237], [0.05499931797385216]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.006892362609505653], [-0.06635904312133789], [0.016340315341949463], [0.024487247690558434]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6691748a8a87d09f2a04ccd61d787fb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[1.301361322402954], [2.505718946456909], [-117.34202575683594], [0.5547717809677124]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2bdf48ae6973d8a4487566beb1fae511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cd5f48fd0a0dc75571e19fa8fddc71b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3ce80ebd0d4a4ba74bc78d438add8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3ce80ebd0d4a4ba74bc78d438add8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3ce80ebd0d4a4ba74bc78d438add8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3ce80ebd0d4a4ba74bc78d438add8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3ce80ebd0d4a4ba74bc78d438add8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3ce80ebd0d4a4ba74bc78d438add8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3ce80ebd0d4a4ba74bc78d438add8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3ce80ebd0d4a4ba74bc78d438add8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3ce80ebd0d4a4ba74bc78d438add8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3ce80ebd0d4a4ba74bc78d438add8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3ce80ebd0d4a4ba74bc78d438add8f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15aa17eea687877cf988c12a2c059c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_358bc8c1eb98b6e964028dcc0f56dec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cd5f48fd0a0dc75571e19fa8fddc71b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e76620bcf0d595eda1ab5ed69cc6b979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a325bb76560ca99b51e435c001af07da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1505b6ac78a5252a396889880f9fe89b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ad43aa4f62461977c2e13b01428b556c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4905618131160736, 0.4396417438983917, 0.38613632321357727, 0.30494746565818787], [0.48475587368011475, 0.07337385416030884, 0.278387188911438, 0.3098049461841583], [0.29222574830055237, 0.4004157781600952, 0.19416268169879913, 0.0951867327094078], [0.29222574830055237, 0.4004157781600952, 0.19416268169879913, 0.0951867327094078], [0.26826053857803345, 0.21500079333782196, 0.11182934045791626, 0.27236950397491455]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.3478876054286957, 0.4785013496875763, 0.05671229213476181, 0.23212681710720062], [0.25702235102653503, 0.06689565628767014, 0.20180051028728485, 0.24164846539497375], [0.3288941979408264, 0.4574975371360779, 0.30252569913864136, 0.07800009846687317], [0.3288941979408264, 0.4574975371360779, 0.30252569913864136, 0.07800009846687317], [0.40340694785118103, 0.1492466926574707, 0.29067420959472656, 0.32166600227355957]], dtype='float32').reshape([5, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5eedd941575a8025b1f6142513fa4c9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_127472bdb620c0be0bd3b5b64ded6cf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba0043b2de27b0634ba412b7f49006eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba0043b2de27b0634ba412b7f49006eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba0043b2de27b0634ba412b7f49006eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba0043b2de27b0634ba412b7f49006eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba0043b2de27b0634ba412b7f49006eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba0043b2de27b0634ba412b7f49006eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba0043b2de27b0634ba412b7f49006eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba0043b2de27b0634ba412b7f49006eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba0043b2de27b0634ba412b7f49006eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba0043b2de27b0634ba412b7f49006eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba0043b2de27b0634ba412b7f49006eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fadf83c28010ecd5bd8925902fc6620e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0906bc96c54199c50c23766f6e035159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_127472bdb620c0be0bd3b5b64ded6cf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_266177bacdbc6e0a792e0f610c2e9b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15494532883167267, 0.2111409604549408, 0.16940054297447205, 0.07837673276662827], [0.22056446969509125, 0.49623557925224304, 0.30928391218185425, 0.13696753978729248], [0.1627940982580185, 0.29646024107933044, 0.267007052898407, 0.4014988839626312], [0.15494532883167267, 0.2111409604549408, 0.16940054297447205, 0.07837673276662827], [0.0711192935705185, 0.08745941519737244, 0.3294025659561157, 0.2732483148574829], [0.333394855260849, 0.2548178434371948, 0.49965474009513855, 0.017060736194252968], [0.0711192935705185, 0.08745941519737244, 0.3294025659561157, 0.2732483148574829]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.46240881085395813, 0.32869765162467957, 0.24423934519290924, 0.25821471214294434], [0.06261877715587616, 0.2492293268442154, 0.43208566308021545, 0.3304005563259125], [0.17018108069896698, 0.49496328830718994, 0.39770105481147766, 0.4668426513671875], [0.46240881085395813, 0.32869765162467957, 0.24423934519290924, 0.25821471214294434], [0.13235357403755188, 0.3676305413246155, 0.23703737556934357, 0.31373870372772217], [0.06706759333610535, 0.2564274072647095, 0.31626176834106445, 0.42102324962615967], [0.13235357403755188, 0.3676305413246155, 0.23703737556934357, 0.31373870372772217]], dtype='float32').reshape([7, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c1607c584dc7e104775c4f773344f40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()