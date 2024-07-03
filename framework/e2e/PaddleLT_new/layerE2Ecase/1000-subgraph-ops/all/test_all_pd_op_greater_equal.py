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
class PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5(InstanceTrait, paddle.nn.Layer):
    
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


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5aa21b40dba082d687900f8b40a2d5a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4607bd6ba16ce5934e14062789eb8588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[150], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]



class PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f(InstanceTrait, paddle.nn.Layer):
    
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


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4a05b2072d997c925aa324a281302dff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_792d7455a824c7392c6944aebe619781(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6895d72a4bfc2c0043fa5c04e496334d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5aa21b40dba082d687900f8b40a2d5a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d08b27bae0f76d20cdebc66b49d1fcd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_485de10fa88fcc76d10c894598a469ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b597d089e4e09918a5310abab1079ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]



class PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6(InstanceTrait, paddle.nn.Layer):
    
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


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_921558bacead5ba679ce21afdc8a1bd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80520f12fa22c7098d11930a55d5f639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9113264c690faa3ec489a2052ac97f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df7d67657d45d4d7e43c1ed18c3d59aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fbce2b1b5dddfd9538879a131db406e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fbce2b1b5dddfd9538879a131db406e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_921558bacead5ba679ce21afdc8a1bd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_26e8b9cc2b85cd7d2c4df2d641a9d61d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_26e8b9cc2b85cd7d2c4df2d641a9d61d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d8169ade3fc5e87ce07c2828ac6ed5cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e619bbcc72b75d7f67b830492ced2be8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c4a914c9215c7aa655bb89a8edd4ccc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2204], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37bbd02c31f8b9e8ffcd0344a4896e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_535270b8fcdba88fcd2be02495e5bd44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_14949125407ba4fae02cdb2c547f0a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a7d5dcb5ce0daab7e92782c8346437e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[551], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2aefe02276f2bcc1b60b90d3907188da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc203a22cfcd795016cd6715e2f6f41d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b591f968bf5638f5e400843c0e7fe330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_420e7cfc5efb055854c09f4819c5b038(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[8816], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9113264c690faa3ec489a2052ac97f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9113264c690faa3ec489a2052ac97f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9113264c690faa3ec489a2052ac97f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b38acc2d2cb88da196b7b2590ce7d984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9113264c690faa3ec489a2052ac97f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9113264c690faa3ec489a2052ac97f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9113264c690faa3ec489a2052ac97f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b38acc2d2cb88da196b7b2590ce7d984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d8169ade3fc5e87ce07c2828ac6ed5cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_26e8b9cc2b85cd7d2c4df2d641a9d61d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc203a22cfcd795016cd6715e2f6f41d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5aa21b40dba082d687900f8b40a2d5a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b591f968bf5638f5e400843c0e7fe330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9113264c690faa3ec489a2052ac97f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_14949125407ba4fae02cdb2c547f0a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5a0614f3837fdd51e6bda58163c72e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9113264c690faa3ec489a2052ac97f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d8169ade3fc5e87ce07c2828ac6ed5cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f0502f15f5a558d62037b221e8b15d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5aa21b40dba082d687900f8b40a2d5a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4607bd6ba16ce5934e14062789eb8588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[150], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4a05b2072d997c925aa324a281302dff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_792d7455a824c7392c6944aebe619781(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6895d72a4bfc2c0043fa5c04e496334d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5aa21b40dba082d687900f8b40a2d5a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d08b27bae0f76d20cdebc66b49d1fcd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_485de10fa88fcc76d10c894598a469ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b597d089e4e09918a5310abab1079ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]



class PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb(InstanceTrait, paddle.nn.Layer):
    
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


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_31d33715bee9264d37422ab472ed2761(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80520f12fa22c7098d11930a55d5f639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_735eeb9a8f590800a6c2a0a3a24b0db5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2e41c3f9fe51096d6ed170fa777b14bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fbce2b1b5dddfd9538879a131db406e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fbce2b1b5dddfd9538879a131db406e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_31d33715bee9264d37422ab472ed2761(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8f7d0cacac8377065c9e3c5f62d9b449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8f7d0cacac8377065c9e3c5f62d9b449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e6cfe4c937ac575ff2e2c8d8ccbfa50f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e619bbcc72b75d7f67b830492ced2be8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c4a914c9215c7aa655bb89a8edd4ccc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2204], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37bbd02c31f8b9e8ffcd0344a4896e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_535270b8fcdba88fcd2be02495e5bd44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_14949125407ba4fae02cdb2c547f0a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a7d5dcb5ce0daab7e92782c8346437e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[551], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2aefe02276f2bcc1b60b90d3907188da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc203a22cfcd795016cd6715e2f6f41d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b591f968bf5638f5e400843c0e7fe330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_420e7cfc5efb055854c09f4819c5b038(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[8816], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_735eeb9a8f590800a6c2a0a3a24b0db5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_735eeb9a8f590800a6c2a0a3a24b0db5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_735eeb9a8f590800a6c2a0a3a24b0db5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_27895dec038ac7405e867db6a17fc591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_735eeb9a8f590800a6c2a0a3a24b0db5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_735eeb9a8f590800a6c2a0a3a24b0db5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_735eeb9a8f590800a6c2a0a3a24b0db5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_27895dec038ac7405e867db6a17fc591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e6cfe4c937ac575ff2e2c8d8ccbfa50f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8f7d0cacac8377065c9e3c5f62d9b449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc203a22cfcd795016cd6715e2f6f41d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5aa21b40dba082d687900f8b40a2d5a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b591f968bf5638f5e400843c0e7fe330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_735eeb9a8f590800a6c2a0a3a24b0db5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_14949125407ba4fae02cdb2c547f0a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4f6ab85d3eea44de84df5344115bc5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5a0614f3837fdd51e6bda58163c72e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44f41e46ec5241199cb80a7dfbec605f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_735eeb9a8f590800a6c2a0a3a24b0db5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e6cfe4c937ac575ff2e2c8d8ccbfa50f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34ed1bc1596f92d723c48b3ca0a124fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()