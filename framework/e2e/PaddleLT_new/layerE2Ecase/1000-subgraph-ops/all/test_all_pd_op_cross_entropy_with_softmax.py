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


    class TestPrimitiveOp_6feabb187ecb9ae0f72dd3a5dca9bee2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1723, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_6feabb187ecb9ae0f72dd3a5dca9bee2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1723, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_b42ca9427e66d56c77dfeb26a1b1770c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5498, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_b42ca9427e66d56c77dfeb26a1b1770c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5498, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_fa657f9d4d0f52065f0aa5c062f2064c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17b4a614998c7fe70554f9b432f8e8d1
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1759, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_fa657f9d4d0f52065f0aa5c062f2064c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17b4a614998c7fe70554f9b432f8e8d1
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1759, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_ec31523df108f34fa6079d9b8a7e15e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1538, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ec31523df108f34fa6079d9b8a7e15e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1538, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_e8b5e1dcbe1cd299e37600683d996094(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2135, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_e8b5e1dcbe1cd299e37600683d996094(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2135, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_dba41c093740fad7b42c07d6a440f2ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4590, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_dba41c093740fad7b42c07d6a440f2ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4590, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_840a11dcbfb350ebdae42531e30d4b28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1042, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_840a11dcbfb350ebdae42531e30d4b28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1042, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_33f1453e04001e987734181c8e35abd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2339, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_33f1453e04001e987734181c8e35abd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2339, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_202aac1e42cfb45602b73a24d024b27b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3063, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_202aac1e42cfb45602b73a24d024b27b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3063, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_3df4f06ff55bcf6f9c7369b2516cc1db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3822, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_3df4f06ff55bcf6f9c7369b2516cc1db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3822, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_9581418f43b5da602c017c2304c41a9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2057, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9581418f43b5da602c017c2304c41a9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2057, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9c0bf5a01b1e7012b560caf1393465e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4189, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9c0bf5a01b1e7012b560caf1393465e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4189, 4, 1], dtype='int64'),
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


    
    class PrimitiveOp_515a511ebe3635bad73eca289d03eae3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1723, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[1723, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f360d1cb5dc431d5e5d9fc0f11840fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_515a511ebe3635bad73eca289d03eae3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1723, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_0f360d1cb5dc431d5e5d9fc0f11840fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_515a511ebe3635bad73eca289d03eae3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1723, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_1e687f17384b4ca894d3ad0d9f0fb116(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5498, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[5498, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4a81c4b1858907d1d48d16e614129339(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e687f17384b4ca894d3ad0d9f0fb116
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5498, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_4a81c4b1858907d1d48d16e614129339(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e687f17384b4ca894d3ad0d9f0fb116
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5498, 4, 1], dtype='int64'),
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


    
    class PrimitiveOp_77201b0727cb1be3f32c5ef6e90e881a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1759, 4, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[1759, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d21b4b4424baaf98c77909b2bcffed72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77201b0727cb1be3f32c5ef6e90e881a
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1759, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d21b4b4424baaf98c77909b2bcffed72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77201b0727cb1be3f32c5ef6e90e881a
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1759, 4, 1], dtype='int64'),
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


    
    class PrimitiveOp_46d13ff89d3951da99a042936da5c7bd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1538, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[1538, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce3fd7106aac387b850dd7fb9bde6355(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d13ff89d3951da99a042936da5c7bd
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1538, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ce3fd7106aac387b850dd7fb9bde6355(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d13ff89d3951da99a042936da5c7bd
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1538, 4, 1], dtype='int64'),
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


    
    class PrimitiveOp_40d12a54b60606c34bc84f0a3533a5ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2135, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[2135, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e4571721ce7a3c04a1216cf2827114c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40d12a54b60606c34bc84f0a3533a5ce
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2135, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9e4571721ce7a3c04a1216cf2827114c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40d12a54b60606c34bc84f0a3533a5ce
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2135, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_dc8d76e204bd87b06f302e5f4a791adc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4590, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[4590, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_42c00c41915be001c2ded5fed9739b71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc8d76e204bd87b06f302e5f4a791adc
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4590, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_42c00c41915be001c2ded5fed9739b71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc8d76e204bd87b06f302e5f4a791adc
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4590, 4, 1], dtype='int64'),
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


    
    class PrimitiveOp_0e27a628bb8456a8c97c12db1c02d1ac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1042, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[1042, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2ca6ab09164472f100e3ef0de9dd105b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e27a628bb8456a8c97c12db1c02d1ac
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1042, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_2ca6ab09164472f100e3ef0de9dd105b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e27a628bb8456a8c97c12db1c02d1ac
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1042, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_bbeec1c859b2450092a4bbc6f6fc735a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2339, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[2339, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c7818829a61cc2fa4231c02871f075c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbeec1c859b2450092a4bbc6f6fc735a
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2339, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_4c7818829a61cc2fa4231c02871f075c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbeec1c859b2450092a4bbc6f6fc735a
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2339, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_b045ab100d12ca5e25c490880cd4b815(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3063, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[3063, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f13c488ac689b22ceeca0408e0cd6dec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b045ab100d12ca5e25c490880cd4b815
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3063, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_f13c488ac689b22ceeca0408e0cd6dec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b045ab100d12ca5e25c490880cd4b815
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3063, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_fab4f71ac6ec5d5ff601f350073fdd61(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3822, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[3822, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff7a38e2ab97282cbb8458acf8ba795f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fab4f71ac6ec5d5ff601f350073fdd61
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3822, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ff7a38e2ab97282cbb8458acf8ba795f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fab4f71ac6ec5d5ff601f350073fdd61
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3822, 4, 1], dtype='int64'),
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


    
    class PrimitiveOp_038b8c6e62772ac0d9abf2735a6c668b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2057, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[2057, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41d99783aa62e6a5cf48aaf3c69ce6b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_038b8c6e62772ac0d9abf2735a6c668b
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2057, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_41d99783aa62e6a5cf48aaf3c69ce6b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_038b8c6e62772ac0d9abf2735a6c668b
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2057, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_5e322885cbb5ff6fb32d136ba2a1ac4b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4189, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[4189, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_348e0a93d8dc94afd5c8444c9f877383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e322885cbb5ff6fb32d136ba2a1ac4b
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4189, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_348e0a93d8dc94afd5c8444c9f877383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e322885cbb5ff6fb32d136ba2a1ac4b
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4189, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_e61b68fd04d626127e12c7af29d9ca3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1723, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_e61b68fd04d626127e12c7af29d9ca3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1723, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_f2d5c0ea690e274efc1b7b20ed595a05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5498, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_f2d5c0ea690e274efc1b7b20ed595a05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5498, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_0d1a11cb63506bb5c3dd57dcc4f225dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1759, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_0d1a11cb63506bb5c3dd57dcc4f225dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1759, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_cc106c7c47e2f1b5d189d848af5a20e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1538, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_cc106c7c47e2f1b5d189d848af5a20e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1538, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_d1d513400906726e89067fc8652c8213(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2135, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d1d513400906726e89067fc8652c8213(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2135, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_52b8c9405ac114e2f1df22dfa9fc1897(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4590, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_52b8c9405ac114e2f1df22dfa9fc1897(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4590, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d9d815fc7aff718545d2c072b6fc0610(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 81], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_0fb054c7a7c7e275976fad3838e01cb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1042, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_0fb054c7a7c7e275976fad3838e01cb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1042, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_bf5eb9314a3d87e24af1c04cdec376bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2339, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_bf5eb9314a3d87e24af1c04cdec376bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2339, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9ff3dec0ed67c73fe1e7269f349f0271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3063, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9ff3dec0ed67c73fe1e7269f349f0271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3063, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_1d986f62e0b1a23ee8469bbb667c0c29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3822, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_1d986f62e0b1a23ee8469bbb667c0c29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3822, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_94b40d6f59f345750c6c386007fe2bff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2057, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_94b40d6f59f345750c6c386007fe2bff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2057, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_469c84882ad23c56029dcada49e9e0ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4189, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_469c84882ad23c56029dcada49e9e0ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4189, 4, 1], dtype='int64'),
            ]


    

if __name__ == '__main__':
    unittest.main()