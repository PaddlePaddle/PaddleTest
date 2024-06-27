import os
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
    class PrimitiveOp_597fdb24947ccd20ec94cc941b253b7a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e04cc275d780f0ec650cbc1e4777ecbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_597fdb24947ccd20ec94cc941b253b7a
        def get_inputs(self):
            return [
                paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([16, 1]),
            ]


    class TestPrimitiveOp_304f07ed288e9c3d3b86d464de5b4cd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_597fdb24947ccd20ec94cc941b253b7a
        def get_inputs(self):
            return [
                paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([16, 1]),
            ]


    
    class PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3c7b0603e328376e74d06ae9f0d36c2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1696, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_3c7b0603e328376e74d06ae9f0d36c2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1696, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_a56eabb5079d74ed3ff756a2beb6c62f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5517, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_a56eabb5079d74ed3ff756a2beb6c62f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5517, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_3569b9b8f27615efd316defe733bf192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_597fdb24947ccd20ec94cc941b253b7a
        def get_inputs(self):
            return [
                paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_3569b9b8f27615efd316defe733bf192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_597fdb24947ccd20ec94cc941b253b7a
        def get_inputs(self):
            return [
                paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_17b4a614998c7fe70554f9b432f8e8d1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c43ec8fb7ed0a886f2856ebe9674e3ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17b4a614998c7fe70554f9b432f8e8d1
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1794, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_c43ec8fb7ed0a886f2856ebe9674e3ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17b4a614998c7fe70554f9b432f8e8d1
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1794, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_2ae8e484ac1b7f26daa1920d8fc89822(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_597fdb24947ccd20ec94cc941b253b7a
        def get_inputs(self):
            return [
                paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([24, 1]),
            ]


    class TestPrimitiveOp_12cffda0c0d0684b47d4c89a338a6176(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_597fdb24947ccd20ec94cc941b253b7a
        def get_inputs(self):
            return [
                paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([24, 1]),
            ]


    class TestPrimitiveOp_6143e976786c9bd8a1dab69561c17143(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1504, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_6143e976786c9bd8a1dab69561c17143(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1504, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_c6dac4b274dbef3a0e54599088f764db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_597fdb24947ccd20ec94cc941b253b7a
        def get_inputs(self):
            return [
                paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0]], dtype='int64').reshape([4, 1]),
            ]


    class TestPrimitiveOp_96f6f49715d94c1eccc0495ffab23946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_597fdb24947ccd20ec94cc941b253b7a
        def get_inputs(self):
            return [
                paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1]], dtype='int64').reshape([4, 1]),
            ]


    class TestPrimitiveOp_02a8de6332ba81bf3149677570996308(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2039, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_02a8de6332ba81bf3149677570996308(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2039, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_5f959f95ca445c44a89b0230d95b8dc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4584, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_5f959f95ca445c44a89b0230d95b8dc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4584, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_9e2206a4cc1e2ebf70af99691dd0fdfb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cb427a641760c743fd9625a52a2624df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2206a4cc1e2ebf70af99691dd0fdfb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 81], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_612b347ad3be4f37a8181ff1f928627b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1071, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_612b347ad3be4f37a8181ff1f928627b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1071, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_dc154a810b6c83ae033fcbc5b5179200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2370, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_dc154a810b6c83ae033fcbc5b5179200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2370, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_75f0de8fdddb97ba77d59f2496978049(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2993, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_75f0de8fdddb97ba77d59f2496978049(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2993, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_5be608fc5033dceb69f16fc32742c2f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3832, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_5be608fc5033dceb69f16fc32742c2f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3832, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_3a7926058434d3575e841fbadf9fce11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_597fdb24947ccd20ec94cc941b253b7a
        def get_inputs(self):
            return [
                paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([20, 1]),
            ]


    class TestPrimitiveOp_1f9d1bab202360410f7a8396fecf51e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_597fdb24947ccd20ec94cc941b253b7a
        def get_inputs(self):
            return [
                paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([20, 1]),
            ]


    class TestPrimitiveOp_2c0f04504c816fad721b7f32278a1c50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2206a4cc1e2ebf70af99691dd0fdfb
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 21], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_851f53f2e3114d71d41730e4c15dea6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1995, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_851f53f2e3114d71d41730e4c15dea6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1995, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_803e0d14d0508c353737b60728657c96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4181, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_803e0d14d0508c353737b60728657c96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4181, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_df48011dfedb88b989dfdaf6bd2b302f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[16, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1742ed4b48f1d13d18b10b27899d4bbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df48011dfedb88b989dfdaf6bd2b302f
        def get_inputs(self):
            return [
                paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([16, 1]),
            ]


    class TestPrimitiveOp_49b632a2898b8006f1a49ff98dc4d761(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df48011dfedb88b989dfdaf6bd2b302f
        def get_inputs(self):
            return [
                paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([16, 1]),
            ]


    
    class PrimitiveOp_6eccb02ae91a36ce4870897bfddd3016(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1696, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[1696, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d6662cec09560eb75d840be87772829f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6eccb02ae91a36ce4870897bfddd3016
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1696, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d6662cec09560eb75d840be87772829f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6eccb02ae91a36ce4870897bfddd3016
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1696, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_bb31bd92c02da915b59df227a2a32593(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5517, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[5517, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6dda053ab4acc4e2d2522103fff54b37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb31bd92c02da915b59df227a2a32593
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5517, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_6dda053ab4acc4e2d2522103fff54b37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb31bd92c02da915b59df227a2a32593
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5517, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_a9fda622a967b3f9fc08228f21d90916(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[36, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[36, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_56feb1bc420ed29ac66f927903b0fbca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9fda622a967b3f9fc08228f21d90916
        def get_inputs(self):
            return [
                paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_56feb1bc420ed29ac66f927903b0fbca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9fda622a967b3f9fc08228f21d90916
        def get_inputs(self):
            return [
                paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_2a9e242fa7945b8cd6caec8b9394907d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1794, 4, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[1794, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae3f84f55157f82b8fc9981af7e3b0c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a9e242fa7945b8cd6caec8b9394907d
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1794, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ae3f84f55157f82b8fc9981af7e3b0c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a9e242fa7945b8cd6caec8b9394907d
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1794, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_cffa9a55ab2fc757a05f858e3d9efc9b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_039c4c6baae8a1c52245f812d35d6ac0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cffa9a55ab2fc757a05f858e3d9efc9b
        def get_inputs(self):
            return [
                paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([24, 1]),
            ]


    class TestPrimitiveOp_c13ac0a994129bd5f454a10081747a5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cffa9a55ab2fc757a05f858e3d9efc9b
        def get_inputs(self):
            return [
                paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([24, 1]),
            ]


    
    class PrimitiveOp_bd2a774c5300a4eb7f6d35ad8030d5ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1504, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[1504, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e3551aaa73ca5e9158c686115ceee4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd2a774c5300a4eb7f6d35ad8030d5ea
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1504, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_4e3551aaa73ca5e9158c686115ceee4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd2a774c5300a4eb7f6d35ad8030d5ea
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1504, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_ffbfaf3e416c0524f45ef32af68a46f6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ee711d15221e16857a71981a20399fd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ffbfaf3e416c0524f45ef32af68a46f6
        def get_inputs(self):
            return [
                paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0]], dtype='int64').reshape([4, 1]),
            ]


    class TestPrimitiveOp_d53374f056242c6323eeae6f24f4b2a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ffbfaf3e416c0524f45ef32af68a46f6
        def get_inputs(self):
            return [
                paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1]], dtype='int64').reshape([4, 1]),
            ]


    
    class PrimitiveOp_ce44ecd562b3613858583bdf3c9bc2bd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2039, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[2039, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6b59ef63b24844e8cae4a41caacf02bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce44ecd562b3613858583bdf3c9bc2bd
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2039, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_6b59ef63b24844e8cae4a41caacf02bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce44ecd562b3613858583bdf3c9bc2bd
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2039, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_aef9c00c90d8613795148272db4de834(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4584, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[4584, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c57e17c96da14f5b40dfc7aabcfc179(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aef9c00c90d8613795148272db4de834
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4584, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_4c57e17c96da14f5b40dfc7aabcfc179(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aef9c00c90d8613795148272db4de834
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4584, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_6068e88559cdae015ab8f97442c78779(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2434, 81], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2434, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c40982ef5927aadf96c5f6e1b50180a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6068e88559cdae015ab8f97442c78779
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 81], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_11f4ada3009c1bd878bc443cee19d3d8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1071, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[1071, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_836f802e58e738ed85f3a645376c2198(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11f4ada3009c1bd878bc443cee19d3d8
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1071, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_836f802e58e738ed85f3a645376c2198(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11f4ada3009c1bd878bc443cee19d3d8
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1071, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_af08c1237f01005e683b64a52ed4d433(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2370, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[2370, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2de57281c101f4bee5ead386336b1ed3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af08c1237f01005e683b64a52ed4d433
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2370, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_2de57281c101f4bee5ead386336b1ed3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af08c1237f01005e683b64a52ed4d433
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2370, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_332b9031884b99f6a7c3cb285b3e8e7c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2993, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[2993, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e9ef878baddb709289e249588b685871(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_332b9031884b99f6a7c3cb285b3e8e7c
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2993, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_e9ef878baddb709289e249588b685871(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_332b9031884b99f6a7c3cb285b3e8e7c
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2993, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_c5b351081249b4d0f5b8f2a5591819c1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3832, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[3832, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b85d26e99951e4fe3969a5d04d9146d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b351081249b4d0f5b8f2a5591819c1
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3832, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_2b85d26e99951e4fe3969a5d04d9146d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b351081249b4d0f5b8f2a5591819c1
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3832, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_d67dc3fbdd87517a9189613a45006f1e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[20, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6f044684206660311f63ec92a93f5788(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d67dc3fbdd87517a9189613a45006f1e
        def get_inputs(self):
            return [
                paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([20, 1]),
            ]


    class TestPrimitiveOp_204e925e1c25799384b040d5d7a5f0ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d67dc3fbdd87517a9189613a45006f1e
        def get_inputs(self):
            return [
                paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([20, 1]),
            ]


    
    class PrimitiveOp_2c8474258a730421bd45245d5c739cbe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8732, 21], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8732, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f84a82ecd53e46c67668b9b53be57f13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c8474258a730421bd45245d5c739cbe
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 21], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_a437c31507798e9aee8cfe87f74eee8d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1995, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[1995, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4dabef22b624f7584f5307bd8a485fc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a437c31507798e9aee8cfe87f74eee8d
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1995, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_4dabef22b624f7584f5307bd8a485fc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a437c31507798e9aee8cfe87f74eee8d
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1995, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_84d1f8042414188d249e0d544ea1f708(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4181, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[4181, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_587ce23d62614b444e2cbafc857f4229(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84d1f8042414188d249e0d544ea1f708
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4181, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_587ce23d62614b444e2cbafc857f4229(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84d1f8042414188d249e0d544ea1f708
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4181, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_02bb574ac5113782413ad22ffab504a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f63a76dd4d8cc0f6f058120cae312d7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02bb574ac5113782413ad22ffab504a3
        def get_inputs(self):
            return [
                paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([16, 1]),
            ]


    class TestPrimitiveOp_9e558a475a865c60e5a86d1fa98f58da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02bb574ac5113782413ad22ffab504a3
        def get_inputs(self):
            return [
                paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([16, 1]),
            ]


    
    class PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_93bb077cc943731600c5d6650b8a959a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1696, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_93bb077cc943731600c5d6650b8a959a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1696, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9b0965835abfece9cdb9ccac5f7862e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5517, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9b0965835abfece9cdb9ccac5f7862e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5517, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9ece1413944a71aa372f4badc8d4dee6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02bb574ac5113782413ad22ffab504a3
        def get_inputs(self):
            return [
                paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9ece1413944a71aa372f4badc8d4dee6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02bb574ac5113782413ad22ffab504a3
        def get_inputs(self):
            return [
                paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_18b024e71e074389e568cb4c72d33a26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1794, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_18b024e71e074389e568cb4c72d33a26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1794, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9501bf3a3c1c449a69b573b88a8e7317(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02bb574ac5113782413ad22ffab504a3
        def get_inputs(self):
            return [
                paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([24, 1]),
            ]


    class TestPrimitiveOp_c0024331c5e1a14bdf05751b3861cb45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02bb574ac5113782413ad22ffab504a3
        def get_inputs(self):
            return [
                paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([24, 1]),
            ]


    class TestPrimitiveOp_8fca2ee72338674c1eaf031ba1fd3fca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1504, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_8fca2ee72338674c1eaf031ba1fd3fca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1504, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_b287a4f858246a73a1cb9a8e7a29cf92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02bb574ac5113782413ad22ffab504a3
        def get_inputs(self):
            return [
                paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0]], dtype='int64').reshape([4, 1]),
            ]


    class TestPrimitiveOp_7bae21ac4e2a4a48a31fdd659c824f61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02bb574ac5113782413ad22ffab504a3
        def get_inputs(self):
            return [
                paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1]], dtype='int64').reshape([4, 1]),
            ]


    class TestPrimitiveOp_89156ebb38d096df177db055fd6d4af1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2039, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_89156ebb38d096df177db055fd6d4af1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2039, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_49618c2753190c42b88f22feca8b3590(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4584, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_49618c2753190c42b88f22feca8b3590(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4584, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d9d815fc7aff718545d2c072b6fc0610(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 81], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_917f2a11c09b63062ddde3ea65b054b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1071, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_917f2a11c09b63062ddde3ea65b054b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1071, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_6f78958ba2d3dbe8c475bb2f0bbe9171(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2370, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_6f78958ba2d3dbe8c475bb2f0bbe9171(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2370, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_c9b73778924de7e6c78cc9e5ee7ff573(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2993, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_c9b73778924de7e6c78cc9e5ee7ff573(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2993, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_e42ca0f6b0fd0ed2af2611254d0c9f8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3832, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_e42ca0f6b0fd0ed2af2611254d0c9f8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3832, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_1ee141e496ed84afd314d1d8c683c3e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02bb574ac5113782413ad22ffab504a3
        def get_inputs(self):
            return [
                paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([20, 1]),
            ]


    class TestPrimitiveOp_58cdd2d93d246e4521e5aace95e7c999(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02bb574ac5113782413ad22ffab504a3
        def get_inputs(self):
            return [
                paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([20, 1]),
            ]


    class TestPrimitiveOp_04dbc041105f314db3661505caa00a09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 21], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_268111ca84a99828349ba872dd7be02c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1995, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_268111ca84a99828349ba872dd7be02c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1995, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d0a964d0a156e280a9a52c4845211138(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4181, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d0a964d0a156e280a9a52c4845211138(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4181, 4, 1], dtype='int64'),
            ]


    

if __name__ == '__main__':
    unittest.main()