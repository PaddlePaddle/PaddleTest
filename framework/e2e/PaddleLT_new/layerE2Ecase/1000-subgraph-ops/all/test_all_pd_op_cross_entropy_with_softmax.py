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
    class PrimitiveOp_dbf57852f3f11537120d28b6b7aa533f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_474232c6ddcdbf49d5ecf4119eca988b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbf57852f3f11537120d28b6b7aa533f
        def get_inputs(self):
            return [
                paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([16, 1]),
            ]


    class TestPrimitiveOp_d3a7cf8c619ed24384bab84eb9973dbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbf57852f3f11537120d28b6b7aa533f
        def get_inputs(self):
            return [
                paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([16, 1]),
            ]


    
    class PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d76d85d43b55b670aca20771f1c89b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1827, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d76d85d43b55b670aca20771f1c89b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1827, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_a41c696eaa623fa64211be5034f34f73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5514, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_a41c696eaa623fa64211be5034f34f73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5514, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_125f27a147f8dfacbf4a578a08a848dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbf57852f3f11537120d28b6b7aa533f
        def get_inputs(self):
            return [
                paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_125f27a147f8dfacbf4a578a08a848dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbf57852f3f11537120d28b6b7aa533f
        def get_inputs(self):
            return [
                paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_b56eee07984520844750e36746179150(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_29b2e5005adcfa676b3b3e0cf1de3c55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b56eee07984520844750e36746179150
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1799, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_29b2e5005adcfa676b3b3e0cf1de3c55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b56eee07984520844750e36746179150
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1799, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_b46a8b9f28bc2fd980c4b7ee2942aa8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbf57852f3f11537120d28b6b7aa533f
        def get_inputs(self):
            return [
                paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([24, 1]),
            ]


    class TestPrimitiveOp_54f6ed7a34dc6169ed33933b6f68212c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbf57852f3f11537120d28b6b7aa533f
        def get_inputs(self):
            return [
                paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([24, 1]),
            ]


    class TestPrimitiveOp_dd5db2a7326406d91a3a4100fc002c9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1503, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_dd5db2a7326406d91a3a4100fc002c9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1503, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_47c96e45d73cafb397e747ef64ec4c04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbf57852f3f11537120d28b6b7aa533f
        def get_inputs(self):
            return [
                paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0]], dtype='int64').reshape([4, 1]),
            ]


    class TestPrimitiveOp_4a89fecf359cd90d34a74c2f9bca2d6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbf57852f3f11537120d28b6b7aa533f
        def get_inputs(self):
            return [
                paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1]], dtype='int64').reshape([4, 1]),
            ]


    class TestPrimitiveOp_c9271df15690994678ed27ccdd935905(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2077, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_c9271df15690994678ed27ccdd935905(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2077, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d5b480f3f00e6350bca619ac7d87c43f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4628, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d5b480f3f00e6350bca619ac7d87c43f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4628, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_f316cc14feb991a0eb775cb50e7d0cf7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_01ed6afa358a28b7c1c5c6b2b81edb6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f316cc14feb991a0eb775cb50e7d0cf7
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 81], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_3879ffc3c2f4fb535a9fb6526351774c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1101, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_3879ffc3c2f4fb535a9fb6526351774c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1101, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ebdc5c02bc27d8e632ae6a91dd2cbc06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2361, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ebdc5c02bc27d8e632ae6a91dd2cbc06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2361, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d190fd7ed498d3d2ac6647c1189c744f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3061, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d190fd7ed498d3d2ac6647c1189c744f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3061, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9e31543e4f75ff588c33a6da4f671012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3799, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9e31543e4f75ff588c33a6da4f671012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3799, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_c8e1d02351c34f92e96a9b2cbc872c4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbf57852f3f11537120d28b6b7aa533f
        def get_inputs(self):
            return [
                paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([20, 1]),
            ]


    class TestPrimitiveOp_138a820854169c308ad3a28f1c0a1bc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbf57852f3f11537120d28b6b7aa533f
        def get_inputs(self):
            return [
                paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([20, 1]),
            ]


    class TestPrimitiveOp_28e4899e091fdf3dde8d0bef8970bc70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f316cc14feb991a0eb775cb50e7d0cf7
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 21], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_001c718e7d3fac2f6e89603efca60510(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2088, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_001c718e7d3fac2f6e89603efca60510(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2088, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d7b2960c5f3273b6208526d18db82682(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4270, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d7b2960c5f3273b6208526d18db82682(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96c979bf49c9f55da6ec26dba84d9de7
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4270, 4, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_03f79bf5451c290d071abd9bc3a46270(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_97737b1bf095db7fa60fd43ae39da398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03f79bf5451c290d071abd9bc3a46270
        def get_inputs(self):
            return [
                paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([16, 1]),
            ]


    class TestPrimitiveOp_c1f7eff0603a24b56367a54924f840a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03f79bf5451c290d071abd9bc3a46270
        def get_inputs(self):
            return [
                paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([16, 1]),
            ]


    
    class PrimitiveOp_6db75651b45a704c5064d098dbd20566(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_adb422e400ecf7ae5f9ad85e93a30f1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1827, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_adb422e400ecf7ae5f9ad85e93a30f1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1827, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_bac6f06e45706eb4aecc607de09342c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5514, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_bac6f06e45706eb4aecc607de09342c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[5514, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_df12dd8d527e18995a4defce171eacb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03f79bf5451c290d071abd9bc3a46270
        def get_inputs(self):
            return [
                paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_df12dd8d527e18995a4defce171eacb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03f79bf5451c290d071abd9bc3a46270
        def get_inputs(self):
            return [
                paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_466f09430ebf4472ce8fc9b3fa41d7aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1799, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_466f09430ebf4472ce8fc9b3fa41d7aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4, 19], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1799, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_7e0b0cfd5cce77fb8ec270f5999220c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03f79bf5451c290d071abd9bc3a46270
        def get_inputs(self):
            return [
                paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([24, 1]),
            ]


    class TestPrimitiveOp_32e1a8669857ac9cbdacdfed9bae0a69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03f79bf5451c290d071abd9bc3a46270
        def get_inputs(self):
            return [
                paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([24, 1]),
            ]


    class TestPrimitiveOp_69cbe7c47313891fe090be77d39f83ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1503, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_69cbe7c47313891fe090be77d39f83ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1503, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_8cb17642b07148617c6f5bef07f90165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03f79bf5451c290d071abd9bc3a46270
        def get_inputs(self):
            return [
                paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0]], dtype='int64').reshape([4, 1]),
            ]


    class TestPrimitiveOp_77a94973cf8bd762a8b79d81962206b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03f79bf5451c290d071abd9bc3a46270
        def get_inputs(self):
            return [
                paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1]], dtype='int64').reshape([4, 1]),
            ]


    class TestPrimitiveOp_a0799e5004632cafe5f63684280deec1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2077, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_a0799e5004632cafe5f63684280deec1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2077, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_2a739aa73bb37937d9f170e24bde8334(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4628, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_2a739aa73bb37937d9f170e24bde8334(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4628, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_cc4fc88738e7418afc24a72f64f4ca6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 81], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_8e74ed24d86a228133fb55deb477ec5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1101, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_8e74ed24d86a228133fb55deb477ec5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1101, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_4181f2de143829a472b2adce665bced0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2361, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_4181f2de143829a472b2adce665bced0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2361, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_0fb3da98407609078f8a99933c1ca8e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3061, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_0fb3da98407609078f8a99933c1ca8e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3061, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ad9296fd884e4898f8adc823a852b9d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3799, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ad9296fd884e4898f8adc823a852b9d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3799, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_3f8fc0b0d5d961254a41b85811a85591(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03f79bf5451c290d071abd9bc3a46270
        def get_inputs(self):
            return [
                paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([20, 1]),
            ]


    class TestPrimitiveOp_3f6b90ef256925623503c759c48ad7fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03f79bf5451c290d071abd9bc3a46270
        def get_inputs(self):
            return [
                paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([20, 1]),
            ]


    class TestPrimitiveOp_e90f2c21e2669c2d0102299832f7168e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 21], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_4cacea4db95fbb7514da75410eb6d0bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2088, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_4cacea4db95fbb7514da75410eb6d0bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2088, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_8344c24618e194894cb53c52abe0aaba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4270, 4, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_8344c24618e194894cb53c52abe0aaba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db75651b45a704c5064d098dbd20566
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4, 17], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4270, 4, 1], dtype='int64'),
            ]


    

if __name__ == '__main__':
    unittest.main()