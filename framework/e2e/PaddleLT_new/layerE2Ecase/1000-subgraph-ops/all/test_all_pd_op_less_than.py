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
    class PrimitiveOp_e277f7c2b62889c61e22e197e427eda8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3e6d98670a04d05eee81469fb32a7ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e277f7c2b62889c61e22e197e427eda8
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_bccdcfc05c85b3a8650ae1181071d796(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e277f7c2b62889c61e22e197e427eda8
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 < input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ce7abdadabed98abeda43c8a7756a11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_8f36711851d360be30657f97db8a588f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[150], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ef598e39301d549de1889d0715fe3173(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_852eab45f4f7ee36bd8a3c69712b9163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f219a11bef84e985b6d8aa7fd6590f53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c86251dbd1bbb357c751ea27e9b49b1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_dee48dedbf3e1bd131060f1730043ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e277f7c2b62889c61e22e197e427eda8
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_572effd660e1098a155e548ddc104c57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2204], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_380d85000a7788fd399973ea577fa172(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5f74b0a2381e67a7ed46262c5d45ebfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[551], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_57851e8abe44f78ce872220e12d1c7a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_189fdab7881841831e5edad962d3542f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7f9fa5f8150bb1a1344f1f5a59d7f26e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[8816], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_adc479372eb604b869c41c79ce7cd1a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e277f7c2b62889c61e22e197e427eda8
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_3585f82f7147f7cf327fc7d2053343f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e277f7c2b62889c61e22e197e427eda8
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_a4afafc49731eb9aff818e123f473953(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_8ce7abdadabed98abeda43c8a7756a11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_06d3040596b097e9f5a65c710d453244(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f2d120c7fc8b9c6d5fec99875c00f96b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3e6d98670a04d05eee81469fb32a7ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e277f7c2b62889c61e22e197e427eda8
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_bccdcfc05c85b3a8650ae1181071d796(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e277f7c2b62889c61e22e197e427eda8
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_8ce7abdadabed98abeda43c8a7756a11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_8f36711851d360be30657f97db8a588f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[150], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ef598e39301d549de1889d0715fe3173(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_852eab45f4f7ee36bd8a3c69712b9163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f219a11bef84e985b6d8aa7fd6590f53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c86251dbd1bbb357c751ea27e9b49b1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_dee48dedbf3e1bd131060f1730043ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e277f7c2b62889c61e22e197e427eda8
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_572effd660e1098a155e548ddc104c57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2204], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_380d85000a7788fd399973ea577fa172(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5f74b0a2381e67a7ed46262c5d45ebfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[551], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_57851e8abe44f78ce872220e12d1c7a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_189fdab7881841831e5edad962d3542f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7f9fa5f8150bb1a1344f1f5a59d7f26e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[8816], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_adc479372eb604b869c41c79ce7cd1a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e277f7c2b62889c61e22e197e427eda8
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_3585f82f7147f7cf327fc7d2053343f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e277f7c2b62889c61e22e197e427eda8
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_a4afafc49731eb9aff818e123f473953(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_8ce7abdadabed98abeda43c8a7756a11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_06d3040596b097e9f5a65c710d453244(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(81, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f2d120c7fc8b9c6d5fec99875c00f96b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e146cd3053f07ba6628a7e19c9769f6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(80, dtype='int64').reshape([]),
            ]


    

if __name__ == '__main__':
    unittest.main()