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
    class PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_054d823571ba377fd481083fa5b91786(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[300, 1], dtype='int32'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_977ed882f7e0bd1e5f562defba51f3f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_1b9ae9d6304dffcf73015a005daf939f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_14ba2a139305a02788ea25c1cc6b8c3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b9ae9d6304dffcf73015a005daf939f
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4e77cd628126b7b3ff4ec801a509f952(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[100, 1], dtype='int32'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_3b300a0bd3428a14261cb3da8f188ecb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_77c7dcda7eab1fc8ab0d927c7bee8391(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b300a0bd3428a14261cb3da8f188ecb
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.randint(low=0, high=3, shape=[2100], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_225c1866fd74b2f9e583971677a6010e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d7f9b584db7717689ced02cb170cb539(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_225c1866fd74b2f9e583971677a6010e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3770636320114136, 0.3660375475883484, 0.49091559648513794, 0.1786925196647644]], dtype='float32').reshape([1, 4]),
                paddle.randint(low=0, high=3, shape=[2100], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5f06f5bf2f5eee13c93f2244891f45ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_897b227820220ff732ae5e4ff50c9f4a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b73ac9c2c1806beee0f509ac3059674(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_897b227820220ff732ae5e4ff50c9f4a
        def get_inputs(self):
            return [
                paddle.uniform([185691], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_53c0fbacdabaa9fedc271871a97e71bd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b5b4e861e592f9b7a8468728f51c3c43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53c0fbacdabaa9fedc271871a97e71bd
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b30aae0d91c151bf381f675b9806036(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4], [2]], dtype='int64').reshape([8, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2b30aae0d91c151bf381f675b9806036(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4], [2]], dtype='int64').reshape([8, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_14ba2a139305a02788ea25c1cc6b8c3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b9ae9d6304dffcf73015a005daf939f
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_977ed882f7e0bd1e5f562defba51f3f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6de5275e971e143faf3fee3467170c99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b300a0bd3428a14261cb3da8f188ecb
        def get_inputs(self):
            return [
                paddle.to_tensor([9, 5], dtype='int32').reshape([2]),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4ff20d58f9275271020f2661a9c6be19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b300a0bd3428a14261cb3da8f188ecb
        def get_inputs(self):
            return [
                paddle.to_tensor([6, 0, 2, 8, 9, 6, 2, 5, 4, 0, 2, 4, 2, 2, 3, 5, 2, 4, 4, 1, 0], dtype='int32').reshape([21]),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ff5fb665f657c8653e9975de2f75c1a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_897b227820220ff732ae5e4ff50c9f4a
        def get_inputs(self):
            return [
                paddle.uniform([242991], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_656dbdcd83909c2abdc26add59889b77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53c0fbacdabaa9fedc271871a97e71bd
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_295aa489ccea283a1b369aef49acfcf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [0], [6], [1], [5]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_295aa489ccea283a1b369aef49acfcf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [0], [6], [1], [5]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bc9e22908de56de2a6b1daeb5c2803d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4c87723adee4785d3c311714b0a73df8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b300a0bd3428a14261cb3da8f188ecb
        def get_inputs(self):
            return [
                paddle.to_tensor([8, 5], dtype='int32').reshape([2]),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a55fa3cac53f1158ca0cbe9702e91205(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_897b227820220ff732ae5e4ff50c9f4a
        def get_inputs(self):
            return [
                paddle.uniform([171888], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f7df1ecc3b86d5c21478c4e2c9720969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53c0fbacdabaa9fedc271871a97e71bd
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e6f8ec455f86bbb38d6956fbc165cd5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e6f8ec455f86bbb38d6956fbc165cd5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_3f5bc7ac60fd6bca3efa46e641f649ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_28ae8c687f21f4e5b5449d9b225ce6c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f5bc7ac60fd6bca3efa46e641f649ee
        def get_inputs(self):
            return [
                paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5]], dtype='int32').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a55fa3cac53f1158ca0cbe9702e91205(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_897b227820220ff732ae5e4ff50c9f4a
        def get_inputs(self):
            return [
                paddle.uniform([171888], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f7df1ecc3b86d5c21478c4e2c9720969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53c0fbacdabaa9fedc271871a97e71bd
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_27aa1c1d29fbca2bfa3aaae4c7e7e5ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1], [3], [3]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_27aa1c1d29fbca2bfa3aaae4c7e7e5ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1], [3], [3]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_fb9d1217a0a56cf1c27a452c51e3942f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca
        def get_inputs(self):
            return [
                paddle.uniform([3, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2]], dtype='int32').reshape([3, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1bbab630324bd84663d5f08977d9daf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_897b227820220ff732ae5e4ff50c9f4a
        def get_inputs(self):
            return [
                paddle.uniform([217413], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_775d6e8e3a76e1735bdcfa6166f042dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53c0fbacdabaa9fedc271871a97e71bd
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0a7cc51257ef0970fe58e598498a0a93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[103, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0a7cc51257ef0970fe58e598498a0a93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[103, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_14ba2a139305a02788ea25c1cc6b8c3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b9ae9d6304dffcf73015a005daf939f
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ce2ac2d28f99c28be5eb2a50a98c0fef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f5bc7ac60fd6bca3efa46e641f649ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_981f960f8e29565879f9dd651142526d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_52629f1c12059394daf3b9bfe129f89b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b300a0bd3428a14261cb3da8f188ecb
        def get_inputs(self):
            return [
                paddle.to_tensor([6, 6], dtype='int32').reshape([2]),
                paddle.randint(low=0, high=3, shape=[3549], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_fd9987abde41facc3893dcfeba0a2c11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_225c1866fd74b2f9e583971677a6010e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.24678689241409302, 0.28349795937538147, 0.03993966057896614, 0.11532165110111237], [0.4213424623012543, 0.055486343801021576, 0.0963839516043663, 0.37215298414230347]], dtype='float32').reshape([2, 4]),
                paddle.randint(low=0, high=3, shape=[3549], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_27e02dff432d286f98ee97becbf4ae4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca
        def get_inputs(self):
            return [
                paddle.uniform([7, 64, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ce2ac2d28f99c28be5eb2a50a98c0fef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f5bc7ac60fd6bca3efa46e641f649ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_706ac9462b1b40a1ffae5927b47e9a73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_897b227820220ff732ae5e4ff50c9f4a
        def get_inputs(self):
            return [
                paddle.uniform([86970], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_20f314cd4288fa98a251c34127e013d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53c0fbacdabaa9fedc271871a97e71bd
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9760d5acfb41af3589439f6f22d6d0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [5], [1], [0], [0], [1]], dtype='int64').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9760d5acfb41af3589439f6f22d6d0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [5], [1], [0], [0], [1]], dtype='int64').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_855ac2d8dae251ed70ed1ebb50982a03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_897b227820220ff732ae5e4ff50c9f4a
        def get_inputs(self):
            return [
                paddle.uniform([205923], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f3a304e46a470cd8da8e0768ea59889a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53c0fbacdabaa9fedc271871a97e71bd
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7c694abec6c49d49e22265645c41694e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [0], [8], [4], [1]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7c694abec6c49d49e22265645c41694e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [0], [8], [4], [1]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_898d5a7e96e718c7a00ec6d89db4bc66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_897b227820220ff732ae5e4ff50c9f4a
        def get_inputs(self):
            return [
                paddle.uniform([153450], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2c26b2710d9b8fd0f393a0f15d0f4532(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53c0fbacdabaa9fedc271871a97e71bd
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9c26a37c845d9df21017c4efcef6de64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1], [7], [4], [8], [3]], dtype='int64').reshape([10, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9c26a37c845d9df21017c4efcef6de64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1], [7], [4], [8], [3]], dtype='int64').reshape([10, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_298582c9faf13629fc3b6a73b0ec84f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca
        def get_inputs(self):
            return [
                paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4]], dtype='int32').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_58d4bc03fedaac354c47606e7fc36024(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b300a0bd3428a14261cb3da8f188ecb
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.randint(low=0, high=3, shape=[4116], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_75f00da1dc10f7256a1b90b19c41ee07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_225c1866fd74b2f9e583971677a6010e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.24421477317810059, 0.2911122441291809, 0.44505247473716736, 0.11270521581172943]], dtype='float32').reshape([1, 4]),
                paddle.randint(low=0, high=3, shape=[4116], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bc9e22908de56de2a6b1daeb5c2803d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_65c6b491d040a64bf2ccd966d9739ad9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_897b227820220ff732ae5e4ff50c9f4a
        def get_inputs(self):
            return [
                paddle.uniform([113061], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8d920b77092fb7102f99f542baebd56b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53c0fbacdabaa9fedc271871a97e71bd
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_de0b9329eceb2cd6b28f16760a5bdadf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[2], [6], [7], [8]], dtype='int64').reshape([4, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_de0b9329eceb2cd6b28f16760a5bdadf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[2], [6], [7], [8]], dtype='int64').reshape([4, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bc9e22908de56de2a6b1daeb5c2803d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ce2ac2d28f99c28be5eb2a50a98c0fef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f5bc7ac60fd6bca3efa46e641f649ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_708495f319ec9698b3d0d5142020a4f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_897b227820220ff732ae5e4ff50c9f4a
        def get_inputs(self):
            return [
                paddle.uniform([123783], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b931f3e032699b43615612c89769d0a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53c0fbacdabaa9fedc271871a97e71bd
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_480df2c7a5cc29f87695a4012ac90bee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[84, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_480df2c7a5cc29f87695a4012ac90bee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[84, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_054d823571ba377fd481083fa5b91786(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[300, 1], dtype='int32'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1b73ac9c2c1806beee0f509ac3059674(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_897b227820220ff732ae5e4ff50c9f4a
        def get_inputs(self):
            return [
                paddle.uniform([185691], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b5b4e861e592f9b7a8468728f51c3c43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53c0fbacdabaa9fedc271871a97e71bd
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_da2eafaf6dd98f9cbdadf261e55b1bdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_da2eafaf6dd98f9cbdadf261e55b1bdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_298582c9faf13629fc3b6a73b0ec84f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca
        def get_inputs(self):
            return [
                paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4]], dtype='int32').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_898d5a7e96e718c7a00ec6d89db4bc66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_897b227820220ff732ae5e4ff50c9f4a
        def get_inputs(self):
            return [
                paddle.uniform([153450], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2c26b2710d9b8fd0f393a0f15d0f4532(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53c0fbacdabaa9fedc271871a97e71bd
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8d214afe0f02930b0336fdcf7f7fd340(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1]], dtype='int64').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8d214afe0f02930b0336fdcf7f7fd340(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1]], dtype='int64').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[49, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c18d4a58253524d16f04683c912cfb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c18d4a58253524d16f04683c912cfb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c18d4a58253524d16f04683c912cfb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c18d4a58253524d16f04683c912cfb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c18d4a58253524d16f04683c912cfb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c18d4a58253524d16f04683c912cfb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c18d4a58253524d16f04683c912cfb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c18d4a58253524d16f04683c912cfb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c18d4a58253524d16f04683c912cfb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c18d4a58253524d16f04683c912cfb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c18d4a58253524d16f04683c912cfb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c18d4a58253524d16f04683c912cfb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c18d4a58253524d16f04683c912cfb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c18d4a58253524d16f04683c912cfb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c18d4a58253524d16f04683c912cfb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c18d4a58253524d16f04683c912cfb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b4413368b454dcf207fe821f92ae9ad
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_2ab51a4639964f12aded0697aaca92b3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c5e052f6d87ef7548fb873aaccae31b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ab51a4639964f12aded0697aaca92b3
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1c5e052f6d87ef7548fb873aaccae31b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ab51a4639964f12aded0697aaca92b3
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_74ca57dd87ce0f3b838294318eb1cf29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ab51a4639964f12aded0697aaca92b3
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_74ca57dd87ce0f3b838294318eb1cf29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ab51a4639964f12aded0697aaca92b3
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_590a635a14710666ea78c795b521e8d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ce2ac2d28f99c28be5eb2a50a98c0fef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f5bc7ac60fd6bca3efa46e641f649ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_04fb5c839743a8c032592f519fa757ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b300a0bd3428a14261cb3da8f188ecb
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 2, 3, 5, 2, 4, 4, 1, 0, 6, 8, 6, 0, 6, 9, 3, 4, 9, 4, 0, 0, 7, 8, 6, 1, 9, 3], dtype='int32').reshape([27]),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7a308b7547127ddade3d0ad43c3952d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f5bc7ac60fd6bca3efa46e641f649ee
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4e77cd628126b7b3ff4ec801a509f952(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[100, 1], dtype='int32'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ce2ac2d28f99c28be5eb2a50a98c0fef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f5bc7ac60fd6bca3efa46e641f649ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_3e259d7decb3ca27556d853bdd8e11ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_897b227820220ff732ae5e4ff50c9f4a
        def get_inputs(self):
            return [
                paddle.uniform([220968], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4a1cf2ed26335eea2f278c4f0bb9b55a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53c0fbacdabaa9fedc271871a97e71bd
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_02bdbcfeab21adb76ef61cf0e4f8a558(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [5], [2], [2], [8]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_02bdbcfeab21adb76ef61cf0e4f8a558(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [5], [2], [2], [8]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_3ad634bd2427967d4de34569952fc7b2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_025098d26bc22531e417d718ff42c4cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2, 2, 2, 1, 1, 0, 2, 0, 2, 0, 0, 0, 2, 0, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d31690afb34c122bb7f0dfca2176edc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_525d954375c9deed05118b810255b82c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 2, 2, 0, 2, 0, 2, 1, 0, 2, 1, 2, 2, 2, 2, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_34b4f5df625125bafbcac8f246820a37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 1, 2, 1, 2, 1, 1, 2, 0, 2, 2, 0, 1, 0, 0, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0175f97674d596fb5d85ea9f29215af9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 0, 2, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1, 2, 0, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1f0a62fe49542e5d6964f0164440522d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bec2f008652ddaeb84719ea754648a04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1, 1, 0, 2, 2, 2, 1, 2, 0, 0, 2, 0, 0, 0, 0], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_90ee2f28dfaca5956f4e76713b7407e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f70a6449421906dd84b0ff86b50a6b7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 0, 2, 0, 2, 2, 2, 2, 1, 0, 0, 2, 1, 1, 2, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_3d1026db1bc57ae3c9a4690eb069454f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 0, 0, 1, 1, 2, 1, 0, 2, 1, 2, 0, 0, 0, 0, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e623c12552a14c751798e44b8339cc3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1, 0, 2, 1, 1, 0, 2, 1, 1, 1, 1, 1, 1, 2, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_167ce21b702cc4428ed9fb23e08fed9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 0, 2, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6126d03179e91563e8d533ef0a01cf9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 2, 1, 2, 0, 2, 1, 0, 0, 1, 1, 1, 0, 2, 0, 0], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9ba6ced258726c09b9660222ee2be7d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 0, 1, 2, 2, 0, 0, 2, 2, 0, 2, 1, 2, 1, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5c06db281ebe63693bd5f42b00bf871d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1, 0, 1, 0, 0, 1, 2, 2, 0, 2, 1, 0, 1, 1, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bd90e962f6f6a42694044d089abd623b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_528da7dfbea505e9db7fae78ee411bd1
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 0, 0, 2, 1, 2, 1, 1, 0, 1, 2, 0, 0, 1, 0, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b497638af91f4c18bd1101520a070344(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_897b227820220ff732ae5e4ff50c9f4a
        def get_inputs(self):
            return [
                paddle.uniform([185658], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4f6af2494950dfaf500d4a190bdcd2be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53c0fbacdabaa9fedc271871a97e71bd
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_74df0ca5e98abaadeb3aafb6c5ceb522(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [1], [6], [9], [2], [8], [2]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_74df0ca5e98abaadeb3aafb6c5ceb522(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e918b5d05ee6d162ad4324420f18e3b
        def get_inputs(self):
            return [
                paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [1], [6], [9], [2], [8], [2]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bc9e22908de56de2a6b1daeb5c2803d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93e5eada930ee0cf490e880f5cfa2ca
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_28ae8c687f21f4e5b5449d9b225ce6c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f5bc7ac60fd6bca3efa46e641f649ee
        def get_inputs(self):
            return [
                paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5]], dtype='int32').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_9d4d3b8dd38f90648b90d3632b480e4b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 256, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[300, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e8e906d39813b4ad830314855528f65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d4d3b8dd38f90648b90d3632b480e4b
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[300, 1], dtype='int32'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_eae23bef7cd44a3e9293e42a5c020c97(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 256, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[8, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd434386abcd6668f9e517824fe5250a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae23bef7cd44a3e9293e42a5c020c97
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_4ac671979b20e1cc8788c8649a154454(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 256, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d5730711463bf334e55f5150abc61bf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ac671979b20e1cc8788c8649a154454
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_d9ca31354931fdf1935b3db66640d554(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 256, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[100, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_876feaccc35645bc58f6778938a4d8be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ca31354931fdf1935b3db66640d554
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[100, 1], dtype='int32'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_828c5d47c77cc56604b5f80add3d721b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='int32'),
                paddle.static.InputSpec(shape=[2100], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_04c5cfbafc8e84f7eddea2fbf8c331e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_828c5d47c77cc56604b5f80add3d721b
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.randint(low=0, high=3, shape=[2100], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_778f1557495b5a6a019001bd07d57cf8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[2100], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_172f3f420453cc25bf5a9bdf58310463(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_778f1557495b5a6a019001bd07d57cf8
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3770636320114136, 0.3660375475883484, 0.49091559648513794, 0.1786925196647644]], dtype='float32').reshape([1, 4]),
                paddle.randint(low=0, high=3, shape=[2100], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d5730711463bf334e55f5150abc61bf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ac671979b20e1cc8788c8649a154454
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_b20834815a8a52320ec6f89480475397(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[185691], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_743272bee70f215c6c25c70e6a0f53c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b20834815a8a52320ec6f89480475397
        def get_inputs(self):
            return [
                paddle.uniform([185691], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_f2df1edbec63b666a55725bfd834bfc6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[185691], dtype='int32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8c95b88d4b19c9b9015718357275b8ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2df1edbec63b666a55725bfd834bfc6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_af5690daa1becb514229cf584ed80098(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[185691, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[8, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_74fd7f80e0091beeb4ae399064ac7259(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af5690daa1becb514229cf584ed80098
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4], [2]], dtype='int64').reshape([8, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_74fd7f80e0091beeb4ae399064ac7259(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af5690daa1becb514229cf584ed80098
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4], [2]], dtype='int64').reshape([8, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d5730711463bf334e55f5150abc61bf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ac671979b20e1cc8788c8649a154454
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bd434386abcd6668f9e517824fe5250a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae23bef7cd44a3e9293e42a5c020c97
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_82829a63a12eb58d3dfaf9ab9b4a4a1e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2], dtype='int32'),
                paddle.static.InputSpec(shape=[2002], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f7e6c7347f66be0fe21d75f4e12a874d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82829a63a12eb58d3dfaf9ab9b4a4a1e
        def get_inputs(self):
            return [
                paddle.to_tensor([9, 5], dtype='int32').reshape([2]),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_46bfb15314b64e76ebd9600087ed8110(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[21], dtype='int32'),
                paddle.static.InputSpec(shape=[1021], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_08ab0ca0f9d15701cc16f42e60bf8fae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46bfb15314b64e76ebd9600087ed8110
        def get_inputs(self):
            return [
                paddle.to_tensor([6, 0, 2, 8, 9, 6, 2, 5, 4, 0, 2, 4, 2, 2, 3, 5, 2, 4, 4, 1, 0], dtype='int32').reshape([21]),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_8f5bcd058df3d75b30182ded231d091e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[242991], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d856cdf7de6256066b786028d36f966(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f5bcd058df3d75b30182ded231d091e
        def get_inputs(self):
            return [
                paddle.uniform([242991], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_6c7a526408b90d65a187bc6cd859538b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[242991], dtype='int32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf5be41cb098dfaacac23d40b21074e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c7a526408b90d65a187bc6cd859538b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_13650be3b19d59c07013784e773f6c90(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[242991, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[5, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a02507ee55e0e4004614d7bf38107f1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13650be3b19d59c07013784e773f6c90
        def get_inputs(self):
            return [
                paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [0], [6], [1], [5]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a02507ee55e0e4004614d7bf38107f1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13650be3b19d59c07013784e773f6c90
        def get_inputs(self):
            return [
                paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [0], [6], [1], [5]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_7c2898512695fb6a780e3f7fb4ea6b42(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7, 256, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[7, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89669cda0b3e6629358139378093ea05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c2898512695fb6a780e3f7fb4ea6b42
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_aaa0917ab245c854192324f912f0edcb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2], dtype='int32'),
                paddle.static.InputSpec(shape=[1002], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_24360279c1355c8867a2832b236e4042(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aaa0917ab245c854192324f912f0edcb
        def get_inputs(self):
            return [
                paddle.to_tensor([8, 5], dtype='int32').reshape([2]),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_fd6fe4076fa5f5f57230255f43b7d7c2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171888], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_597ed7a2e1daf9bb4c63df0e508744a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd6fe4076fa5f5f57230255f43b7d7c2
        def get_inputs(self):
            return [
                paddle.uniform([171888], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_e0a1beda22eda722abd1808114908488(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171888], dtype='int32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fd599a8128a7cf1c48c1cc27a1cb0d89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0a1beda22eda722abd1808114908488
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_60df212c38d6e4bf68e8e302f3027af5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171888, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[5, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_29333c4d5b448c64ab8f3f9a0652152e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60df212c38d6e4bf68e8e302f3027af5
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_29333c4d5b448c64ab8f3f9a0652152e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60df212c38d6e4bf68e8e302f3027af5
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_a63c5f3da1a2a17c0ec4a8cea51c174b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 256, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[6, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_267aa3df950fdac80e3fa0ad6a0c74ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a63c5f3da1a2a17c0ec4a8cea51c174b
        def get_inputs(self):
            return [
                paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5]], dtype='int32').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_597ed7a2e1daf9bb4c63df0e508744a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd6fe4076fa5f5f57230255f43b7d7c2
        def get_inputs(self):
            return [
                paddle.uniform([171888], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_fd599a8128a7cf1c48c1cc27a1cb0d89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0a1beda22eda722abd1808114908488
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_dd1bb43c5f9075501af5a7d6808f185d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171888, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[7, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_33b42704b572ffda69722febfa27fed6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd1bb43c5f9075501af5a7d6808f185d
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1], [3], [3]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_33b42704b572ffda69722febfa27fed6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd1bb43c5f9075501af5a7d6808f185d
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1], [3], [3]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_a0762be6b409d39d6375d2ca4b61a4e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3, 256, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[3, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_746438ec9ea37b07e6d9ce5d50d9db2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0762be6b409d39d6375d2ca4b61a4e6
        def get_inputs(self):
            return [
                paddle.uniform([3, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2]], dtype='int32').reshape([3, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_b090f9d099672a2e8834528a3fd341e8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[217413], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_097ba86b993ce0f3f79fad2f864fe7d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b090f9d099672a2e8834528a3fd341e8
        def get_inputs(self):
            return [
                paddle.uniform([217413], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_06fef136a335e5ac79fd6ebb15b1661b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[217413], dtype='int32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b393c7aa79a7e5082cbe5ecafcbd2dc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06fef136a335e5ac79fd6ebb15b1661b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_2bf3862843aeabef19f4f9a8a3dc95ed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[217413, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[103, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ebcbd9f51be636d72bd2875cda96dd1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bf3862843aeabef19f4f9a8a3dc95ed
        def get_inputs(self):
            return [
                paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[103, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ebcbd9f51be636d72bd2875cda96dd1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bf3862843aeabef19f4f9a8a3dc95ed
        def get_inputs(self):
            return [
                paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[103, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d5730711463bf334e55f5150abc61bf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ac671979b20e1cc8788c8649a154454
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_13a605866caeb7d98ed9639ff4d986ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ad07707a96919556f84f53c2de98256(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13a605866caeb7d98ed9639ff4d986ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_acdf75233ca59f5e4b896a4a382d556b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_981f960f8e29565879f9dd651142526d
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_c5c1bbee001a56720096ddff2708e2e0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2], dtype='int32'),
                paddle.static.InputSpec(shape=[3549], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5c74b7a11be5cb42c9522183a6c235bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5c1bbee001a56720096ddff2708e2e0
        def get_inputs(self):
            return [
                paddle.to_tensor([6, 6], dtype='int32').reshape([2]),
                paddle.randint(low=0, high=3, shape=[3549], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_2470307f6340f4f788cc081bdcf46a55(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[3549], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a1f986351848021e99e244b226753a20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2470307f6340f4f788cc081bdcf46a55
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.24678689241409302, 0.28349795937538147, 0.03993966057896614, 0.11532165110111237], [0.4213424623012543, 0.055486343801021576, 0.0963839516043663, 0.37215298414230347]], dtype='float32').reshape([2, 4]),
                paddle.randint(low=0, high=3, shape=[3549], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_e0dd5fcc7a0261015d6bd0bb4c37ceca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7, 64, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[7, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fbcd19192998f6780050bb5480f82780(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0dd5fcc7a0261015d6bd0bb4c37ceca
        def get_inputs(self):
            return [
                paddle.uniform([7, 64, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4ad07707a96919556f84f53c2de98256(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13a605866caeb7d98ed9639ff4d986ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_38d5f08380945887459dc349f2158d83(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86970], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a981b3eb88953050606c177a5aae7c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38d5f08380945887459dc349f2158d83
        def get_inputs(self):
            return [
                paddle.uniform([86970], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_594d81547cb09b4999cf5b20844a34c6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86970], dtype='int32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0ad306c7111080930d983544a72bebd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_594d81547cb09b4999cf5b20844a34c6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_d3574f5fc50d2361e8a12eed0f3db051(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86970, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[6, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1ada2f350e41edbea3fa3e3bc770b2c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3574f5fc50d2361e8a12eed0f3db051
        def get_inputs(self):
            return [
                paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [5], [1], [0], [0], [1]], dtype='int64').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1ada2f350e41edbea3fa3e3bc770b2c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3574f5fc50d2361e8a12eed0f3db051
        def get_inputs(self):
            return [
                paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [5], [1], [0], [0], [1]], dtype='int64').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_78b25136cdc5aed63bd1a93bda2ed961(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[205923], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c91aed1e553e4b7451d790f1a148fc07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78b25136cdc5aed63bd1a93bda2ed961
        def get_inputs(self):
            return [
                paddle.uniform([205923], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_710d315c3d2c7cd9dbcda7a220a92c20(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[205923], dtype='int32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_839db762326a079a2826bd05ba8f7b8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_710d315c3d2c7cd9dbcda7a220a92c20
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_ae9afd3bec0c29710d59c18325c079d9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[205923, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[5, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_19faa8471b057e8f82b5540a03f837e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae9afd3bec0c29710d59c18325c079d9
        def get_inputs(self):
            return [
                paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [0], [8], [4], [1]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_19faa8471b057e8f82b5540a03f837e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae9afd3bec0c29710d59c18325c079d9
        def get_inputs(self):
            return [
                paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [0], [8], [4], [1]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_fd42782d2be054a3c59e006da4d483c2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[153450], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a8e0ccefb4fa566d9af58d0e4dc6064c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd42782d2be054a3c59e006da4d483c2
        def get_inputs(self):
            return [
                paddle.uniform([153450], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_e962be2fe9b152ad21a6d9b1f8dd98f4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[153450], dtype='int32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c3f4a42eae671097e722f54b03ce751b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e962be2fe9b152ad21a6d9b1f8dd98f4
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_43eddd7ad1159bf45250187126ae4849(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[153450, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_86d3c2f9da49c705b092e91c7494b541(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43eddd7ad1159bf45250187126ae4849
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1], [7], [4], [8], [3]], dtype='int64').reshape([10, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_86d3c2f9da49c705b092e91c7494b541(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43eddd7ad1159bf45250187126ae4849
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1], [7], [4], [8], [3]], dtype='int64').reshape([10, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_7e6749620d559ef1f2c5df5284cf1c1c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5, 256, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[5, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fd34e65362a2e171abaff1ef1fe1a9ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e6749620d559ef1f2c5df5284cf1c1c
        def get_inputs(self):
            return [
                paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4]], dtype='int32').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_4b26e25c731295d1d1adf23e4ddc80a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='int32'),
                paddle.static.InputSpec(shape=[4116], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c6664f64478473183ed61005e81a2825(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b26e25c731295d1d1adf23e4ddc80a4
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.randint(low=0, high=3, shape=[4116], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_0b078527d7172ba51e84c2bc31a737a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[4116], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b02425e2c91ad34f1824ed50c1294744(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b078527d7172ba51e84c2bc31a737a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.24421477317810059, 0.2911122441291809, 0.44505247473716736, 0.11270521581172943]], dtype='float32').reshape([1, 4]),
                paddle.randint(low=0, high=3, shape=[4116], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_89669cda0b3e6629358139378093ea05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c2898512695fb6a780e3f7fb4ea6b42
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_41adc4e331d7b03bd475f04cebaa6dfa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[113061], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_13759f1dfb8117e100112706e7e321e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41adc4e331d7b03bd475f04cebaa6dfa
        def get_inputs(self):
            return [
                paddle.uniform([113061], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_d8d63b14b2a01d4a51ad18a236a121aa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[113061], dtype='int32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5f756fdfeed93edb2bce066feddb23f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8d63b14b2a01d4a51ad18a236a121aa
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_1e572747c317e2be2ce9f1586682b0d4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[113061, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[4, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4badd2dd86b3b50dd5fb5cc0b4a34ce4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e572747c317e2be2ce9f1586682b0d4
        def get_inputs(self):
            return [
                paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[2], [6], [7], [8]], dtype='int64').reshape([4, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4badd2dd86b3b50dd5fb5cc0b4a34ce4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e572747c317e2be2ce9f1586682b0d4
        def get_inputs(self):
            return [
                paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[2], [6], [7], [8]], dtype='int64').reshape([4, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_89669cda0b3e6629358139378093ea05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c2898512695fb6a780e3f7fb4ea6b42
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4ad07707a96919556f84f53c2de98256(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13a605866caeb7d98ed9639ff4d986ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_101981382321f4c7e0d711304a8b6582(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[123783], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_90ca2a8a55cddd25b4810daab12f4e75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101981382321f4c7e0d711304a8b6582
        def get_inputs(self):
            return [
                paddle.uniform([123783], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_d8b5e671805513ac0687b9b85ea4ea54(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[123783], dtype='int32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d09e04f5264632830c526437dc80c1aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8b5e671805513ac0687b9b85ea4ea54
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_47ac7d0900576a6357aaf2bcd6e3fa1a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[123783, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[84, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_256b93aaea6f664f4e79f5d93555bbf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47ac7d0900576a6357aaf2bcd6e3fa1a
        def get_inputs(self):
            return [
                paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[84, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_256b93aaea6f664f4e79f5d93555bbf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47ac7d0900576a6357aaf2bcd6e3fa1a
        def get_inputs(self):
            return [
                paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[84, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9e8e906d39813b4ad830314855528f65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d4d3b8dd38f90648b90d3632b480e4b
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[300, 1], dtype='int32'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_743272bee70f215c6c25c70e6a0f53c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b20834815a8a52320ec6f89480475397
        def get_inputs(self):
            return [
                paddle.uniform([185691], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8c95b88d4b19c9b9015718357275b8ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2df1edbec63b666a55725bfd834bfc6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_c378c98cd656443172366b3bfce49b23(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[185691, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[7, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4335b1f88d95acb155eea9a2fe6c7532(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378c98cd656443172366b3bfce49b23
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4335b1f88d95acb155eea9a2fe6c7532(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378c98cd656443172366b3bfce49b23
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_fd34e65362a2e171abaff1ef1fe1a9ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e6749620d559ef1f2c5df5284cf1c1c
        def get_inputs(self):
            return [
                paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4]], dtype='int32').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a8e0ccefb4fa566d9af58d0e4dc6064c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd42782d2be054a3c59e006da4d483c2
        def get_inputs(self):
            return [
                paddle.uniform([153450], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c3f4a42eae671097e722f54b03ce751b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e962be2fe9b152ad21a6d9b1f8dd98f4
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_2d07f98a7430cec4bace98a54209ba03(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[153450, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[6, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea14fcdcda68aec39757aa0ed6b9478b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d07f98a7430cec4bace98a54209ba03
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1]], dtype='int64').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ea14fcdcda68aec39757aa0ed6b9478b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d07f98a7430cec4bace98a54209ba03
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1]], dtype='int64').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_7429e64b263c779708b827f3cfb031fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[49, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[49], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae0dbcdbfcba0c83bc7f54473ae6ca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7429e64b263c779708b827f3cfb031fd
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae0dbcdbfcba0c83bc7f54473ae6ca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7429e64b263c779708b827f3cfb031fd
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae0dbcdbfcba0c83bc7f54473ae6ca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7429e64b263c779708b827f3cfb031fd
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae0dbcdbfcba0c83bc7f54473ae6ca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7429e64b263c779708b827f3cfb031fd
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae0dbcdbfcba0c83bc7f54473ae6ca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7429e64b263c779708b827f3cfb031fd
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae0dbcdbfcba0c83bc7f54473ae6ca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7429e64b263c779708b827f3cfb031fd
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae0dbcdbfcba0c83bc7f54473ae6ca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7429e64b263c779708b827f3cfb031fd
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae0dbcdbfcba0c83bc7f54473ae6ca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7429e64b263c779708b827f3cfb031fd
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae0dbcdbfcba0c83bc7f54473ae6ca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7429e64b263c779708b827f3cfb031fd
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae0dbcdbfcba0c83bc7f54473ae6ca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7429e64b263c779708b827f3cfb031fd
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae0dbcdbfcba0c83bc7f54473ae6ca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7429e64b263c779708b827f3cfb031fd
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae0dbcdbfcba0c83bc7f54473ae6ca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7429e64b263c779708b827f3cfb031fd
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae0dbcdbfcba0c83bc7f54473ae6ca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7429e64b263c779708b827f3cfb031fd
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae0dbcdbfcba0c83bc7f54473ae6ca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7429e64b263c779708b827f3cfb031fd
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae0dbcdbfcba0c83bc7f54473ae6ca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7429e64b263c779708b827f3cfb031fd
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ae0dbcdbfcba0c83bc7f54473ae6ca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7429e64b263c779708b827f3cfb031fd
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_08637cee79a26e83fcabf2478c7b2eee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_abd53d6746321698a494af492d611803(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08637cee79a26e83fcabf2478c7b2eee
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_abd53d6746321698a494af492d611803(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08637cee79a26e83fcabf2478c7b2eee
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_717f78cea67469c1a2eab9405b5a21e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ee3ddeaf384a04a5b68bd07ed735c0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_717f78cea67469c1a2eab9405b5a21e7
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8ee3ddeaf384a04a5b68bd07ed735c0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_717f78cea67469c1a2eab9405b5a21e7
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_001bc58e512204a6100c1ec4f9099a0f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fceec337f2b11e3bc9151fd7b265b236(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_001bc58e512204a6100c1ec4f9099a0f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4ad07707a96919556f84f53c2de98256(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13a605866caeb7d98ed9639ff4d986ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_ce8f98d471dffe4b32ea8f2a4ad57d47(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[27], dtype='int32'),
                paddle.static.InputSpec(shape=[1027], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ffacfb9020a643b5a979baa84e79f2f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce8f98d471dffe4b32ea8f2a4ad57d47
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 2, 3, 5, 2, 4, 4, 1, 0, 6, 8, 6, 0, 6, 9, 3, 4, 9, 4, 0, 0, 7, 8, 6, 1, 9, 3], dtype='int32').reshape([27]),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_ceec141da0e0b287dd710b2fdca4bdd7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 256, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[8, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5e7e542b6f8fa5165d10aa0ec34e3f8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ceec141da0e0b287dd710b2fdca4bdd7
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_876feaccc35645bc58f6778938a4d8be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ca31354931fdf1935b3db66640d554
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[100, 1], dtype='int32'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4ad07707a96919556f84f53c2de98256(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13a605866caeb7d98ed9639ff4d986ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e4f5475b8df17bd7ebf5b316424bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795bfcf58b6cd30e7346fe542b9ddc1b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_13929987a3054c99bdd518eb8c64faef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[220968], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_94cbf00079d07c1c6ce98f21b6db5108(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13929987a3054c99bdd518eb8c64faef
        def get_inputs(self):
            return [
                paddle.uniform([220968], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_c7ed15f297b9ab12bdb8055ed1265b1c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[220968], dtype='int32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06b9d9b3abd7040949e77fa25a9a65bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ed15f297b9ab12bdb8055ed1265b1c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_21f76aa34e55a2906a3566bca5586381(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[220968, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[5, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41b7557c190ec4ad0200c573a82209d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21f76aa34e55a2906a3566bca5586381
        def get_inputs(self):
            return [
                paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [5], [2], [2], [8]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_41b7557c190ec4ad0200c573a82209d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21f76aa34e55a2906a3566bca5586381
        def get_inputs(self):
            return [
                paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [5], [2], [2], [8]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_081de5afe8559cc680bde17233cc4a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad634bd2427967d4de34569952fc7b2
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[16], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e06c1b5fc2f6730bb6a52c8678d6685b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2, 2, 2, 1, 1, 0, 2, 0, 2, 0, 0, 0, 2, 0, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b523222f944a75d81c6c7d28b5399a61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_107f9ab2342eb05ab236c3f38a7bb546(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 2, 2, 0, 2, 0, 2, 1, 0, 2, 1, 2, 2, 2, 2, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5a4378bd82d755b06bc8c00b111cd395(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 1, 2, 1, 2, 1, 1, 2, 0, 2, 2, 0, 1, 0, 0, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_49e09deb9b4c07b9a4ee2fc50da83643(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 0, 2, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1, 2, 0, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_933fe8298e976f75e65f9e7704642d81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0374d837697bba459f0f473bbf82b374(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1, 1, 0, 2, 2, 2, 1, 2, 0, 0, 2, 0, 0, 0, 0], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_69aabbb0e979a5583f79c4ce1b125473(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2cd774c72b8062c0d283607e6332e447(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 0, 2, 0, 2, 2, 2, 2, 1, 0, 0, 2, 1, 1, 2, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1e2423f38c97fc2f589d1eb099812227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 0, 0, 1, 1, 2, 1, 0, 2, 1, 2, 0, 0, 0, 0, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8df345e46c8063e61e65c3ad7f30a53d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1, 0, 2, 1, 1, 0, 2, 1, 1, 1, 1, 1, 1, 2, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9a15daf7453af4d0ecba386ba81f50e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 0, 2, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_274ea7d18910b4a976d79181279c2b98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 2, 1, 2, 0, 2, 1, 0, 0, 1, 1, 1, 0, 2, 0, 0], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b3b080e298f7cc720d9aa4c16e057351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 0, 1, 2, 2, 0, 0, 2, 2, 0, 2, 1, 2, 1, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_11c17ad8fcea92a5cf5a3cfc1497ca7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1, 0, 1, 0, 0, 1, 2, 2, 0, 2, 1, 0, 1, 1, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_790b7afbb9fdc0b12f64bd434fb665b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b75c06d92251eaa3f0c6c9dd5e82cd
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 0, 0, 2, 1, 2, 1, 1, 0, 1, 2, 0, 0, 1, 0, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_9604097d5e3cebbefb5d40f017745c9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[185658], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_93ca32c258f1d38cc568a4b75cfa2f9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9604097d5e3cebbefb5d40f017745c9e
        def get_inputs(self):
            return [
                paddle.uniform([185658], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_ddffce671036e7ce5a4dd0d52395ecf4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[185658], dtype='int32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e38844600b926506aba188fee786e6c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ddffce671036e7ce5a4dd0d52395ecf4
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_e3755e1e4c287814b4b23ad61d6c484d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[185658, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[7, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_72c8fabec42387a6af4d947e10f3eec8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3755e1e4c287814b4b23ad61d6c484d
        def get_inputs(self):
            return [
                paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [1], [6], [9], [2], [8], [2]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_72c8fabec42387a6af4d947e10f3eec8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3755e1e4c287814b4b23ad61d6c484d
        def get_inputs(self):
            return [
                paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [1], [6], [9], [2], [8], [2]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_89669cda0b3e6629358139378093ea05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c2898512695fb6a780e3f7fb4ea6b42
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_267aa3df950fdac80e3fa0ad6a0c74ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a63c5f3da1a2a17c0ec4a8cea51c174b
        def get_inputs(self):
            return [
                paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5]], dtype='int32').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_927be18ae3113e0c369b6e2168846d67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[300, 1], dtype='int32'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7784ab59c5c8b9de8d79fcf53d392c3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0159caad17c3e44abb646ffab8d9f768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5e8431ef72fdeb6aacc74b4ad1f36bf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[100, 1], dtype='int32'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_e9264b551643c3ea484952b2723dc50c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2081f8e5730ba2eab9839f124023ee0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9264b551643c3ea484952b2723dc50c
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.randint(low=0, high=3, shape=[2100], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_bba5c10391f831392e6216a52a1d5b28(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_452b2947107fbaa9b8eabf57fc079c55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3770636320114136, 0.3660375475883484, 0.49091559648513794, 0.1786925196647644]], dtype='float32').reshape([1, 4]),
                paddle.randint(low=0, high=3, shape=[2100], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0159caad17c3e44abb646ffab8d9f768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_0f94cef9189347ba11bf98c69504a33d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_048ceea0045ff01f1e6c77aa2636fd68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f94cef9189347ba11bf98c69504a33d
        def get_inputs(self):
            return [
                paddle.uniform([185691], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_d6cb1f2f7c452113f123907a355da01d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3de84e8612c293f5e54857f4d4288f7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6cb1f2f7c452113f123907a355da01d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ef1c5c32d483fd35961981649e7db972(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4], [2]], dtype='int64').reshape([8, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ef1c5c32d483fd35961981649e7db972(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4], [2]], dtype='int64').reshape([8, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0159caad17c3e44abb646ffab8d9f768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7784ab59c5c8b9de8d79fcf53d392c3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b5360d73cbbc3b4d178b7d60d3f43d62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9264b551643c3ea484952b2723dc50c
        def get_inputs(self):
            return [
                paddle.to_tensor([9, 5], dtype='int32').reshape([2]),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6c30c36047e20bde63eac14f1e98ff90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9264b551643c3ea484952b2723dc50c
        def get_inputs(self):
            return [
                paddle.to_tensor([6, 0, 2, 8, 9, 6, 2, 5, 4, 0, 2, 4, 2, 2, 3, 5, 2, 4, 4, 1, 0], dtype='int32').reshape([21]),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a7946e142b0189ecd4bd671d12234166(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f94cef9189347ba11bf98c69504a33d
        def get_inputs(self):
            return [
                paddle.uniform([242991], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9bad19b31d05fc117d8be1b754ba4356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6cb1f2f7c452113f123907a355da01d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5a4d5a9980bae310de2713bd3d3ab0f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [0], [6], [1], [5]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5a4d5a9980bae310de2713bd3d3ab0f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [0], [6], [1], [5]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1b964b643853dae6179bb1feb476f466(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_29149d2ad5d0bcea917fabf2e306c501(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9264b551643c3ea484952b2723dc50c
        def get_inputs(self):
            return [
                paddle.to_tensor([8, 5], dtype='int32').reshape([2]),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9a7551a4e016a607b06ffe10d588ecd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f94cef9189347ba11bf98c69504a33d
        def get_inputs(self):
            return [
                paddle.uniform([171888], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_04b4cc82b156a49243814583ff347509(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6cb1f2f7c452113f123907a355da01d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_be9e81b634764903dc63fbf2d18375b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_be9e81b634764903dc63fbf2d18375b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_adbe034b5ec6099557474b5f9e1e1c8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5]], dtype='int32').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9a7551a4e016a607b06ffe10d588ecd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f94cef9189347ba11bf98c69504a33d
        def get_inputs(self):
            return [
                paddle.uniform([171888], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_04b4cc82b156a49243814583ff347509(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6cb1f2f7c452113f123907a355da01d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9173e2bcbd350105625bc96320b1ee2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1], [3], [3]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9173e2bcbd350105625bc96320b1ee2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1], [3], [3]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ee2ea54b861c7ade5c0c6dfea83a1923(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([3, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2]], dtype='int32').reshape([3, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ed122fdceb51f3319a169ce79fb83ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f94cef9189347ba11bf98c69504a33d
        def get_inputs(self):
            return [
                paddle.uniform([217413], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_017ac44eb7cd6a189bf2e89272f5c29a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6cb1f2f7c452113f123907a355da01d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_fd3f15428453e3e005a72121df409c20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[103, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_fd3f15428453e3e005a72121df409c20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[103, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0159caad17c3e44abb646ffab8d9f768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e140e465ee2bc704dbc7a3bd64ddceb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5465c1196db049a8a3e534cf6c8be14d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_99ebfce844d64002a564b58f0070ef48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9264b551643c3ea484952b2723dc50c
        def get_inputs(self):
            return [
                paddle.to_tensor([6, 6], dtype='int32').reshape([2]),
                paddle.randint(low=0, high=3, shape=[3549], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_832dfd04e3612ef13dd70e2b593fefe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.24678689241409302, 0.28349795937538147, 0.03993966057896614, 0.11532165110111237], [0.4213424623012543, 0.055486343801021576, 0.0963839516043663, 0.37215298414230347]], dtype='float32').reshape([2, 4]),
                paddle.randint(low=0, high=3, shape=[3549], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c783e2328175506c15dfac61793d3ee5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([7, 64, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e140e465ee2bc704dbc7a3bd64ddceb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0c49364d5e8c9e4197b1982b63c0d552(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f94cef9189347ba11bf98c69504a33d
        def get_inputs(self):
            return [
                paddle.uniform([86970], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_45cc83a6c42b137ca105f9beafbc1136(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6cb1f2f7c452113f123907a355da01d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1ccacfe64836b5eae21051da6c2d4bc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [5], [1], [0], [0], [1]], dtype='int64').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1ccacfe64836b5eae21051da6c2d4bc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [5], [1], [0], [0], [1]], dtype='int64').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_70923978ff9f112be82ec38cf2219103(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f94cef9189347ba11bf98c69504a33d
        def get_inputs(self):
            return [
                paddle.uniform([205923], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1eb5f981d5ef3baa90e93940be459b4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6cb1f2f7c452113f123907a355da01d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f876be7f9699df13ae0b50bfb8fc6a7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [0], [8], [4], [1]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f876be7f9699df13ae0b50bfb8fc6a7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [0], [8], [4], [1]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f6f0930aabc31138076be27a443ff7a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f94cef9189347ba11bf98c69504a33d
        def get_inputs(self):
            return [
                paddle.uniform([153450], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e22c4165301c56ecf6dccb399ebeb779(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6cb1f2f7c452113f123907a355da01d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_efda28099a7571a55e8248671a0e7cf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1], [7], [4], [8], [3]], dtype='int64').reshape([10, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_efda28099a7571a55e8248671a0e7cf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1], [7], [4], [8], [3]], dtype='int64').reshape([10, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1402e8d5368428f0389b10b3e72b9dfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4]], dtype='int32').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a2d6257cf95b15b0af561c5d57a3f59d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9264b551643c3ea484952b2723dc50c
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.randint(low=0, high=3, shape=[4116], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9b8afbe4dbdc9cfb7850d93c1984ffb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.24421477317810059, 0.2911122441291809, 0.44505247473716736, 0.11270521581172943]], dtype='float32').reshape([1, 4]),
                paddle.randint(low=0, high=3, shape=[4116], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1b964b643853dae6179bb1feb476f466(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e55530215c5dc31c29e66bee73992e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f94cef9189347ba11bf98c69504a33d
        def get_inputs(self):
            return [
                paddle.uniform([113061], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ec1c7b5170250e12a50ff29b42192374(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6cb1f2f7c452113f123907a355da01d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b6311684210c880b3e6494a506138321(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[2], [6], [7], [8]], dtype='int64').reshape([4, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b6311684210c880b3e6494a506138321(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[2], [6], [7], [8]], dtype='int64').reshape([4, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1b964b643853dae6179bb1feb476f466(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e140e465ee2bc704dbc7a3bd64ddceb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1f7eef10f7294722da3bb4c896166e6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f94cef9189347ba11bf98c69504a33d
        def get_inputs(self):
            return [
                paddle.uniform([123783], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7a7d38d15ec3b9dfa7bbc4ae13764a39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6cb1f2f7c452113f123907a355da01d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2bf46db43a9a7cc8cdafd32ddcd9068a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[84, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2bf46db43a9a7cc8cdafd32ddcd9068a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[84, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_927be18ae3113e0c369b6e2168846d67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[300, 1], dtype='int32'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_048ceea0045ff01f1e6c77aa2636fd68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f94cef9189347ba11bf98c69504a33d
        def get_inputs(self):
            return [
                paddle.uniform([185691], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_3de84e8612c293f5e54857f4d4288f7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6cb1f2f7c452113f123907a355da01d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e31af014168f471c9cf92c70c5794614(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e31af014168f471c9cf92c70c5794614(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1402e8d5368428f0389b10b3e72b9dfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4]], dtype='int32').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f6f0930aabc31138076be27a443ff7a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f94cef9189347ba11bf98c69504a33d
        def get_inputs(self):
            return [
                paddle.uniform([153450], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e22c4165301c56ecf6dccb399ebeb779(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6cb1f2f7c452113f123907a355da01d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d10b9f795c2d2a9ccffc6ccda4d3717d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1]], dtype='int64').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d10b9f795c2d2a9ccffc6ccda4d3717d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1]], dtype='int64').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d58a731bfcc27a1b1bce020abba3d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d58a731bfcc27a1b1bce020abba3d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d58a731bfcc27a1b1bce020abba3d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d58a731bfcc27a1b1bce020abba3d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d58a731bfcc27a1b1bce020abba3d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d58a731bfcc27a1b1bce020abba3d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d58a731bfcc27a1b1bce020abba3d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d58a731bfcc27a1b1bce020abba3d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d58a731bfcc27a1b1bce020abba3d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d58a731bfcc27a1b1bce020abba3d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d58a731bfcc27a1b1bce020abba3d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d58a731bfcc27a1b1bce020abba3d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d58a731bfcc27a1b1bce020abba3d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d58a731bfcc27a1b1bce020abba3d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d58a731bfcc27a1b1bce020abba3d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d58a731bfcc27a1b1bce020abba3d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_30f83ef07ef6c87a7632ae92b6035001(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a670ecd1e5664a77d076ce9b5adb0d57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30f83ef07ef6c87a7632ae92b6035001
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a670ecd1e5664a77d076ce9b5adb0d57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30f83ef07ef6c87a7632ae92b6035001
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1bb2c62f95a968c70d08764ff964ec7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30f83ef07ef6c87a7632ae92b6035001
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1bb2c62f95a968c70d08764ff964ec7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30f83ef07ef6c87a7632ae92b6035001
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_63632094db1b20818264698a8616b3ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e140e465ee2bc704dbc7a3bd64ddceb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d76acfd6449582bbb4e13f7153b45fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9264b551643c3ea484952b2723dc50c
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 2, 3, 5, 2, 4, 4, 1, 0, 6, 8, 6, 0, 6, 9, 3, 4, 9, 4, 0, 0, 7, 8, 6, 1, 9, 3], dtype='int32').reshape([27]),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0364a5748fe26369c97f61b77182486a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5e8431ef72fdeb6aacc74b4ad1f36bf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[100, 1], dtype='int32'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e140e465ee2bc704dbc7a3bd64ddceb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf87c9379fb91c80bb00be09543eff80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2b51a21a521f7850d43401a3e1cf0c2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f94cef9189347ba11bf98c69504a33d
        def get_inputs(self):
            return [
                paddle.uniform([220968], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8cfbf619ad08e4e1bd1397e171f411e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6cb1f2f7c452113f123907a355da01d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7acfb1de1a9c1c4eeb92910a716167df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [5], [2], [2], [8]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7acfb1de1a9c1c4eeb92910a716167df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [5], [2], [2], [8]], dtype='int64').reshape([5, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a19d53e352fcd5422b7059784bee90fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e033ec4963e5adad0f78421949fd4488(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2, 2, 2, 1, 1, 0, 2, 0, 2, 0, 0, 0, 2, 0, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d46f7898aaf43103d3dc2c2b1e5b6916(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_60de999bc930fbcd2c7a5fa00fa36ae9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 2, 2, 0, 2, 0, 2, 1, 0, 2, 1, 2, 2, 2, 2, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e95cf69b843958b8c05c708666890df7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 1, 2, 1, 2, 1, 1, 2, 0, 2, 2, 0, 1, 0, 0, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b1924c3e22d7f9061ac68549c6aa578d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 0, 2, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1, 2, 0, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a7664fcdead401dfed81473355369a55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_36b2473a82f16c69cc5e40014ad4479c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1, 1, 0, 2, 2, 2, 1, 2, 0, 0, 2, 0, 0, 0, 0], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7ccc7ccb2b0dcaa3418ab68de8c5de0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9b1bf28b0bdd6e69dd558c39c3ffdffc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 0, 2, 0, 2, 2, 2, 2, 1, 0, 0, 2, 1, 1, 2, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_32f4027f94f4cf1c94d6487b4e153eec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 0, 0, 1, 1, 2, 1, 0, 2, 1, 2, 0, 0, 0, 0, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_189a232fc9ad3350ef3d00e1855aaafd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1, 0, 2, 1, 1, 0, 2, 1, 1, 1, 1, 1, 1, 2, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4abedf99c76a6f882041174f2e68d8f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 0, 2, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_98f22bb3e89f943ef1afa9a49e6f8670(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 2, 1, 2, 0, 2, 1, 0, 0, 1, 1, 1, 0, 2, 0, 0], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_00f010bfcc805ba9584b7969c16369bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 0, 1, 2, 2, 0, 0, 2, 2, 0, 2, 1, 2, 1, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1b926ef62d4c9ea0a6ec804416786bd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1, 0, 1, 0, 0, 1, 2, 2, 0, 2, 1, 0, 1, 1, 2], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_3930dcab8bda5346bd7ae7b0d987a176(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bba5c10391f831392e6216a52a1d5b28
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 0, 0, 2, 1, 2, 1, 1, 0, 1, 2, 0, 0, 1, 0, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7092ce4fdccc0155605fc32f5d94a756(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f94cef9189347ba11bf98c69504a33d
        def get_inputs(self):
            return [
                paddle.uniform([185658], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c251a765466e84cb3a532913b29fb8b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6cb1f2f7c452113f123907a355da01d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_611aa6dcc101abc41a2365382a3f15ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [1], [6], [9], [2], [8], [2]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_611aa6dcc101abc41a2365382a3f15ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48bc382d59c3625a2f4293e4bbfbc08e
        def get_inputs(self):
            return [
                paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [1], [6], [9], [2], [8], [2]], dtype='int64').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1b964b643853dae6179bb1feb476f466(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_adbe034b5ec6099557474b5f9e1e1c8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69d59a698f3cb4e8df52421e2d2b25c
        def get_inputs(self):
            return [
                paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5]], dtype='int32').reshape([6, 1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()