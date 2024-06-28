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


    class TestPrimitiveOp_05ce2e6db4217ef2c971a0781777b3cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1774, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_05ce2e6db4217ef2c971a0781777b3cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1774, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_8612023f4b1c06c8daae146f80b1a2bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5454, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_8612023f4b1c06c8daae146f80b1a2bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5454, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_5ed26be2d1e7d48d6a5895dda6d93f22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17b4a614998c7fe70554f9b432f8e8d1
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1722, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_5ed26be2d1e7d48d6a5895dda6d93f22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17b4a614998c7fe70554f9b432f8e8d1
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1722, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_c9eb34be7a3f02410d49a9ac1fcfa5f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1518, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_c9eb34be7a3f02410d49a9ac1fcfa5f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1518, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_d0a929ae8d256069ace71b89ee9a2ba2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2133, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d0a929ae8d256069ace71b89ee9a2ba2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2133, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d02c69ee01a3d31a8244cca8e0103881(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4631, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d02c69ee01a3d31a8244cca8e0103881(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4631, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_0851a4dd3b147a724cfafa0f4f76cf29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1039, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_0851a4dd3b147a724cfafa0f4f76cf29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1039, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_5f3fbc23af5170850fa78c020b393797(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2318, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_5f3fbc23af5170850fa78c020b393797(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2318, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ffc2abc7ba01a4f28488cb3b53790733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2961, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ffc2abc7ba01a4f28488cb3b53790733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2961, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_b8569870cc962572757c6e0a0d7db144(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3739, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_b8569870cc962572757c6e0a0d7db144(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3739, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_313f62ea0ecf2efb1d687add0769f14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2013, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_313f62ea0ecf2efb1d687add0769f14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2013, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_1021b5fff23cd340edab2ddbfb42dbf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4177, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_1021b5fff23cd340edab2ddbfb42dbf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0d53e8c5ad8e9ccfbfa4980d14265e2
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4177, 4, 1], dtype='int64'),
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


    
    class PrimitiveOp_737d826a88b8fb743258331b586f3471(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1774, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[1774, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6cfe3459bc252b4c647e19be514fab51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_737d826a88b8fb743258331b586f3471
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1774, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_6cfe3459bc252b4c647e19be514fab51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_737d826a88b8fb743258331b586f3471
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1774, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_78b0e9ff6903da234e554a6861e93fe6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5454, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[5454, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_550248ae5803278a0ced3d6d078d75df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78b0e9ff6903da234e554a6861e93fe6
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5454, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_550248ae5803278a0ced3d6d078d75df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78b0e9ff6903da234e554a6861e93fe6
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5454, 4, 1], dtype='int64'),
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


    
    class PrimitiveOp_79aab98a4b4a7b95ec182902e6340fb2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1722, 4, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[1722, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b1727584b9bb78791724cccec1f2a35c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79aab98a4b4a7b95ec182902e6340fb2
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1722, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_b1727584b9bb78791724cccec1f2a35c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79aab98a4b4a7b95ec182902e6340fb2
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1722, 4, 1], dtype='int64'),
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


    
    class PrimitiveOp_81b7163c9587c2a11f5f2d506b7fde1e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1518, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[1518, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b163468bbc26202612a0e7cf728d16f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81b7163c9587c2a11f5f2d506b7fde1e
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1518, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_2b163468bbc26202612a0e7cf728d16f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81b7163c9587c2a11f5f2d506b7fde1e
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1518, 4, 1], dtype='int64'),
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


    
    class PrimitiveOp_11a958291152f73f9567e57811aa90d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2133, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[2133, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_740b4a33c2175083f5ca4f66ca9b5358(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a958291152f73f9567e57811aa90d7
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2133, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_740b4a33c2175083f5ca4f66ca9b5358(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a958291152f73f9567e57811aa90d7
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2133, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_941c9d6b09cadc57212a984665e6a4bd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4631, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[4631, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c5a0eaa8cca917c3d9c395d333367bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_941c9d6b09cadc57212a984665e6a4bd
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4631, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9c5a0eaa8cca917c3d9c395d333367bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_941c9d6b09cadc57212a984665e6a4bd
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4631, 4, 1], dtype='int64'),
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


    
    class PrimitiveOp_137281441681c386d2c2a99a56e0c3ed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1039, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[1039, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5333d97912ae9761522a7d3374578a40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_137281441681c386d2c2a99a56e0c3ed
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1039, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_5333d97912ae9761522a7d3374578a40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_137281441681c386d2c2a99a56e0c3ed
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1039, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_f1e4a3da2e1b1e1737a7e1a7e8466941(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2318, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[2318, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f37a5c0aa4b793e36f230eb4315b57e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1e4a3da2e1b1e1737a7e1a7e8466941
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2318, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_f37a5c0aa4b793e36f230eb4315b57e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1e4a3da2e1b1e1737a7e1a7e8466941
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2318, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_51bc2d303bc862665a3983d72ce0a38b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2961, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[2961, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c25795fe551186a33a76448ae8a34ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51bc2d303bc862665a3983d72ce0a38b
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2961, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_c25795fe551186a33a76448ae8a34ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51bc2d303bc862665a3983d72ce0a38b
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2961, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_67f273046d01e1a1769085e4f65a71e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3739, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[3739, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e93d24fac50993e746ddd8aa9ecf4df9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67f273046d01e1a1769085e4f65a71e7
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3739, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_e93d24fac50993e746ddd8aa9ecf4df9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67f273046d01e1a1769085e4f65a71e7
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3739, 4, 1], dtype='int64'),
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


    
    class PrimitiveOp_42c8affb97a3d671e4f68904ae938ef2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2013, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[2013, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b888a7c9c7edd007f2de5f051621472b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42c8affb97a3d671e4f68904ae938ef2
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2013, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_b888a7c9c7edd007f2de5f051621472b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42c8affb97a3d671e4f68904ae938ef2
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2013, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_b3a35247b77d83ab6a38d268879a64ff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4177, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[4177, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f338bba1236126e5e4b09ca95bd19b46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3a35247b77d83ab6a38d268879a64ff
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4177, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_f338bba1236126e5e4b09ca95bd19b46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3a35247b77d83ab6a38d268879a64ff
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4177, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_e8d30593598a2eb8a0980fdef9d60fda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1774, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_e8d30593598a2eb8a0980fdef9d60fda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1774, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_3e6bf45241c212f3c5df0b5c26b07f19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5454, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_3e6bf45241c212f3c5df0b5c26b07f19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5454, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_a07d282a3e38e98f6c544d80841b8efd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1722, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_a07d282a3e38e98f6c544d80841b8efd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1722, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_90c8cfcd82e7187e60018295e5ee7196(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1518, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_90c8cfcd82e7187e60018295e5ee7196(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1518, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_6d5d4ee161c4f73dcae2dbc489a2fc7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2133, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_6d5d4ee161c4f73dcae2dbc489a2fc7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2133, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_340b4cb2b6bab316a73a7f9ff9fb9020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4631, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_340b4cb2b6bab316a73a7f9ff9fb9020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4631, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d9d815fc7aff718545d2c072b6fc0610(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 81], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_1ee553e9399a8f6b2740c19e90845ab0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1039, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_1ee553e9399a8f6b2740c19e90845ab0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1039, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_74bd9393ff47ee6537e226b708859046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2318, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_74bd9393ff47ee6537e226b708859046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2318, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_4169477de55fea2a1c2f0a8d4592c0a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2961, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_4169477de55fea2a1c2f0a8d4592c0a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2961, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_03b52d6e6d3a54afccee94008bd5f615(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3739, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_03b52d6e6d3a54afccee94008bd5f615(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3739, 4, 1], dtype='int64'),
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


    class TestPrimitiveOp_8d3740e09fc9bd04ff0529bd9480b1a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2013, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_8d3740e09fc9bd04ff0529bd9480b1a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2013, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_13429f9b4897133519979496cfdc84bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4177, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_13429f9b4897133519979496cfdc84bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0829b4cbf5a6914590abcee891ee1dea
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4177, 4, 1], dtype='int64'),
            ]


    

if __name__ == '__main__':
    unittest.main()