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
    class PrimitiveOp_1a55308501b8885ffd76a09623ec892e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a51d6bf20e957c07261c980fe12d34ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a55308501b8885ffd76a09623ec892e
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[300, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_d60aaa6afd953851a7885a7ad43de831(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a55308501b8885ffd76a09623ec892e
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
            ]


    
    class PrimitiveOp_aa1a01c0fadaab742cfc8fb1700fe535(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_13ee003f7934c4e5287c8884ade4f4ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa1a01c0fadaab742cfc8fb1700fe535
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
            ]


    class TestPrimitiveOp_6c823d3a1bfd53d0c9dc124c80e58983(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a55308501b8885ffd76a09623ec892e
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[100, 1], dtype='int32'),
            ]


    
    class PrimitiveOp_2a31124b695ffeec8cddd1914e89a779(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e91ee1004622206504fe2c744d65a4e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a31124b695ffeec8cddd1914e89a779
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.randint(low=0, high=3, shape=[2100], dtype='int64'),
            ]


    
    class PrimitiveOp_67daa2a0f8c740b40d3474fc7f667456(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0dfd5e2de113a0cc37a699e4f723427b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67daa2a0f8c740b40d3474fc7f667456
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.12885119020938873, 0.11754139512777328, 0.37269124388694763, 0.08517523109912872]], dtype='float32').reshape([1, 4]),
                paddle.randint(low=0, high=3, shape=[2100], dtype='int64'),
            ]


    class TestPrimitiveOp_0f7c551a9ed8351b87868e1f817fcdef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a55308501b8885ffd76a09623ec892e
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
            ]


    
    class PrimitiveOp_d40d7f84f6af5fd0f502190fb463351f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_19d70d4439ed534b6e41ba6f1c55dbd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d40d7f84f6af5fd0f502190fb463351f
        def get_inputs(self):
            return [
                paddle.uniform([185691], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_bd8b90c7bb88e33a8744a034b1b104be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fe9c00e67f93e961193c6d83c0eddcaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd8b90c7bb88e33a8744a034b1b104be
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_1fb198b575821f765d6d519647c28c8e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fe61591601ccbe2b215ce94c074f763c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4], [2]], dtype='int64').reshape([8, 1]),
            ]


    class TestPrimitiveOp_fe61591601ccbe2b215ce94c074f763c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4], [2]], dtype='int64').reshape([8, 1]),
            ]


    class TestPrimitiveOp_13ee003f7934c4e5287c8884ade4f4ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa1a01c0fadaab742cfc8fb1700fe535
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
            ]


    class TestPrimitiveOp_d60aaa6afd953851a7885a7ad43de831(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a55308501b8885ffd76a09623ec892e
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
            ]


    class TestPrimitiveOp_eb237b1677c242c91f26c0aed0629af6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a31124b695ffeec8cddd1914e89a779
        def get_inputs(self):
            return [
                paddle.to_tensor([9, 5], dtype='int32').reshape([2]),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int64'),
            ]


    class TestPrimitiveOp_5fbdd3163c618d371bda1e6f6f7f6b4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a31124b695ffeec8cddd1914e89a779
        def get_inputs(self):
            return [
                paddle.to_tensor([6, 0, 2, 8, 9, 6, 2, 5, 4, 0, 2, 4, 2, 2, 3, 5, 2, 4, 4, 1, 0], dtype='int32').reshape([21]),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int64'),
            ]


    class TestPrimitiveOp_a88d5dad9de3a5593d43be961ec3c627(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d40d7f84f6af5fd0f502190fb463351f
        def get_inputs(self):
            return [
                paddle.uniform([242991], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_06e3c5ad1862e8104870cbc34b225cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd8b90c7bb88e33a8744a034b1b104be
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_bc98c9efb60c79b24902c377a1024488(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [0], [6], [1], [5]], dtype='int64').reshape([5, 1]),
            ]


    class TestPrimitiveOp_bc98c9efb60c79b24902c377a1024488(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [0], [6], [1], [5]], dtype='int64').reshape([5, 1]),
            ]


    class TestPrimitiveOp_9baa9c37023c460a61908df5a4a09fb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a55308501b8885ffd76a09623ec892e
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            ]


    class TestPrimitiveOp_8c3a18c35e61bb11453f4388a6c5ed06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a31124b695ffeec8cddd1914e89a779
        def get_inputs(self):
            return [
                paddle.to_tensor([8, 5], dtype='int32').reshape([2]),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int64'),
            ]


    class TestPrimitiveOp_4c662cae9a90e2e717d3275f2bc178fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d40d7f84f6af5fd0f502190fb463351f
        def get_inputs(self):
            return [
                paddle.uniform([171888], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9717808f9ecdb3f7ee6df9ecc8ea3327(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd8b90c7bb88e33a8744a034b1b104be
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_65597ae1976b969fa9f1f1bfddb26b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1]], dtype='int64').reshape([5, 1]),
            ]


    class TestPrimitiveOp_65597ae1976b969fa9f1f1bfddb26b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1]], dtype='int64').reshape([5, 1]),
            ]


    
    class PrimitiveOp_96aaacd9d57a811336f6298d67d831c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_05059477b9896b10e517b4b72cbf1470(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96aaacd9d57a811336f6298d67d831c7
        def get_inputs(self):
            return [
                paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5]], dtype='int32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_4c662cae9a90e2e717d3275f2bc178fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d40d7f84f6af5fd0f502190fb463351f
        def get_inputs(self):
            return [
                paddle.uniform([171888], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9717808f9ecdb3f7ee6df9ecc8ea3327(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd8b90c7bb88e33a8744a034b1b104be
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_2f13be66c59fed492314731424d23f04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1], [3], [3]], dtype='int64').reshape([7, 1]),
            ]


    class TestPrimitiveOp_2f13be66c59fed492314731424d23f04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1], [3], [3]], dtype='int64').reshape([7, 1]),
            ]


    class TestPrimitiveOp_17354b88845f67ac76e3530c0d26a04c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a55308501b8885ffd76a09623ec892e
        def get_inputs(self):
            return [
                paddle.uniform([3, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2]], dtype='int32').reshape([3, 1]),
            ]


    class TestPrimitiveOp_4da69b8511f387d7a20068e00bd45a3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d40d7f84f6af5fd0f502190fb463351f
        def get_inputs(self):
            return [
                paddle.uniform([217413], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_41e36553d9d39a79f6c46c54afb0ff3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd8b90c7bb88e33a8744a034b1b104be
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_02a718e95c54275226c2fd0917c9792d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[103, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_02a718e95c54275226c2fd0917c9792d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[103, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_13ee003f7934c4e5287c8884ade4f4ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa1a01c0fadaab742cfc8fb1700fe535
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
            ]


    class TestPrimitiveOp_127ed340fc54688ca82da31b21befc2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96aaacd9d57a811336f6298d67d831c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            ]


    
    class PrimitiveOp_7bff159e241d975241034d14e6d35bb2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6739f74f8e7737225a4c76a037553bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bff159e241d975241034d14e6d35bb2
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_d54d1f98233b81dc6b540f9f1d402f8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a31124b695ffeec8cddd1914e89a779
        def get_inputs(self):
            return [
                paddle.to_tensor([6, 6], dtype='int32').reshape([2]),
                paddle.randint(low=0, high=3, shape=[3549], dtype='int64'),
            ]


    class TestPrimitiveOp_6731d0f308df16a5c76cfbeef0dfba6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67daa2a0f8c740b40d3474fc7f667456
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4455467462539673, 0.04675282537937164, 0.2217721790075302, 0.248161181807518], [0.4585568606853485, 0.37466222047805786, 0.08961380273103714, 0.2054985761642456]], dtype='float32').reshape([2, 4]),
                paddle.randint(low=0, high=3, shape=[3549], dtype='int64'),
            ]


    class TestPrimitiveOp_3055c5da6adf67c76e8dd16f0d160968(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a55308501b8885ffd76a09623ec892e
        def get_inputs(self):
            return [
                paddle.uniform([7, 64, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            ]


    class TestPrimitiveOp_127ed340fc54688ca82da31b21befc2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96aaacd9d57a811336f6298d67d831c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_a160d927f4ebc63f8d6aff0e7df038d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d40d7f84f6af5fd0f502190fb463351f
        def get_inputs(self):
            return [
                paddle.uniform([86970], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_460b3812d6333e80e5002bc7efe52848(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd8b90c7bb88e33a8744a034b1b104be
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_f86fd2f345e49eef836455745ade8e08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [5], [1], [0], [0], [1]], dtype='int64').reshape([6, 1]),
            ]


    class TestPrimitiveOp_f86fd2f345e49eef836455745ade8e08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [5], [1], [0], [0], [1]], dtype='int64').reshape([6, 1]),
            ]


    class TestPrimitiveOp_bcb614c36e2288d4133fd45898559feb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d40d7f84f6af5fd0f502190fb463351f
        def get_inputs(self):
            return [
                paddle.uniform([205923], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_16e4610eb983aa4f2dd81c645a0dea23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd8b90c7bb88e33a8744a034b1b104be
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_2e615102abee9e3d38f8793276230940(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [0], [8], [4], [1]], dtype='int64').reshape([5, 1]),
            ]


    class TestPrimitiveOp_2e615102abee9e3d38f8793276230940(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [0], [8], [4], [1]], dtype='int64').reshape([5, 1]),
            ]


    class TestPrimitiveOp_4812cd7e4ce60569d633a80d0348901a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d40d7f84f6af5fd0f502190fb463351f
        def get_inputs(self):
            return [
                paddle.uniform([153450], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ee775768ac7ef6098d38e76b7c5cecc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd8b90c7bb88e33a8744a034b1b104be
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_c614bc94dddd20ca43bfec5d8fca7797(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1], [7], [4], [8], [3]], dtype='int64').reshape([10, 1]),
            ]


    class TestPrimitiveOp_c614bc94dddd20ca43bfec5d8fca7797(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1], [7], [4], [8], [3]], dtype='int64').reshape([10, 1]),
            ]


    class TestPrimitiveOp_031c93b09283cc444936f1ecd261904d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a55308501b8885ffd76a09623ec892e
        def get_inputs(self):
            return [
                paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4]], dtype='int32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_aee15e4d8a791f4c97834f31fea58e45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a31124b695ffeec8cddd1914e89a779
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.randint(low=0, high=3, shape=[4116], dtype='int64'),
            ]


    class TestPrimitiveOp_e201c02b4a0da55789b2c1de8cde3caf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67daa2a0f8c740b40d3474fc7f667456
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15030789375305176, 0.11389752477407455, 0.29615697264671326, 0.26190102100372314]], dtype='float32').reshape([1, 4]),
                paddle.randint(low=0, high=3, shape=[4116], dtype='int64'),
            ]


    class TestPrimitiveOp_9baa9c37023c460a61908df5a4a09fb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a55308501b8885ffd76a09623ec892e
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            ]


    class TestPrimitiveOp_48fdf3091b08a5a23c88df8c6899b409(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d40d7f84f6af5fd0f502190fb463351f
        def get_inputs(self):
            return [
                paddle.uniform([113061], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d26c173d201287cc2bd2cab7de2fd142(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd8b90c7bb88e33a8744a034b1b104be
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_c9b2d0892d4224f5f1b9ab8035ae39f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[2], [6], [7], [8]], dtype='int64').reshape([4, 1]),
            ]


    class TestPrimitiveOp_c9b2d0892d4224f5f1b9ab8035ae39f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[2], [6], [7], [8]], dtype='int64').reshape([4, 1]),
            ]


    class TestPrimitiveOp_9baa9c37023c460a61908df5a4a09fb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a55308501b8885ffd76a09623ec892e
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            ]


    class TestPrimitiveOp_127ed340fc54688ca82da31b21befc2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96aaacd9d57a811336f6298d67d831c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_c5ad5cfd81a7469c9eae360a3346a0c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d40d7f84f6af5fd0f502190fb463351f
        def get_inputs(self):
            return [
                paddle.uniform([123783], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_6b6051675e23288174f63d285950e12d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd8b90c7bb88e33a8744a034b1b104be
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9f6b6257f7ec65ab76e902f6cf865849(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[84, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9f6b6257f7ec65ab76e902f6cf865849(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[84, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_a51d6bf20e957c07261c980fe12d34ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a55308501b8885ffd76a09623ec892e
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[300, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_19d70d4439ed534b6e41ba6f1c55dbd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d40d7f84f6af5fd0f502190fb463351f
        def get_inputs(self):
            return [
                paddle.uniform([185691], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_fe9c00e67f93e961193c6d83c0eddcaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd8b90c7bb88e33a8744a034b1b104be
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_7f822f91b7bb2ccb22f4273d4a34a39c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4]], dtype='int64').reshape([7, 1]),
            ]


    class TestPrimitiveOp_7f822f91b7bb2ccb22f4273d4a34a39c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4]], dtype='int64').reshape([7, 1]),
            ]


    class TestPrimitiveOp_031c93b09283cc444936f1ecd261904d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a55308501b8885ffd76a09623ec892e
        def get_inputs(self):
            return [
                paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4]], dtype='int32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_4812cd7e4ce60569d633a80d0348901a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d40d7f84f6af5fd0f502190fb463351f
        def get_inputs(self):
            return [
                paddle.uniform([153450], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ee775768ac7ef6098d38e76b7c5cecc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd8b90c7bb88e33a8744a034b1b104be
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_44044a99f1e529f3a8801dbd99addd85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1]], dtype='int64').reshape([6, 1]),
            ]


    class TestPrimitiveOp_44044a99f1e529f3a8801dbd99addd85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1]], dtype='int64').reshape([6, 1]),
            ]


    
    class PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[49, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6bd274486e4741b42e5acad3f1e1b6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6bd274486e4741b42e5acad3f1e1b6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6bd274486e4741b42e5acad3f1e1b6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6bd274486e4741b42e5acad3f1e1b6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6bd274486e4741b42e5acad3f1e1b6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6bd274486e4741b42e5acad3f1e1b6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6bd274486e4741b42e5acad3f1e1b6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6bd274486e4741b42e5acad3f1e1b6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6bd274486e4741b42e5acad3f1e1b6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6bd274486e4741b42e5acad3f1e1b6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6bd274486e4741b42e5acad3f1e1b6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6bd274486e4741b42e5acad3f1e1b6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6bd274486e4741b42e5acad3f1e1b6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6bd274486e4741b42e5acad3f1e1b6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6bd274486e4741b42e5acad3f1e1b6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_6bd274486e4741b42e5acad3f1e1b6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41cdf76bf1d3e404398e051d6f41687
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    
    class PrimitiveOp_e98250a819fb0f02ba52e84ae9d7cf21(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 1
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_369960a4fb68c4fe9211e667267c8f4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e98250a819fb0f02ba52e84ae9d7cf21
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_369960a4fb68c4fe9211e667267c8f4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e98250a819fb0f02ba52e84ae9d7cf21
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_8e316967898300aa503488f598d6d33b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e98250a819fb0f02ba52e84ae9d7cf21
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_8e316967898300aa503488f598d6d33b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e98250a819fb0f02ba52e84ae9d7cf21
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_d0cc0082c399e4d7f0e95ffd447c0153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a55308501b8885ffd76a09623ec892e
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_127ed340fc54688ca82da31b21befc2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96aaacd9d57a811336f6298d67d831c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_28ebae9afd03c6401122cfad3e8afe8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a31124b695ffeec8cddd1914e89a779
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 2, 3, 5, 2, 4, 4, 1, 0, 6, 8, 6, 0, 6, 9, 3, 4, 9, 4, 0, 0, 7, 8, 6, 1, 9, 3], dtype='int32').reshape([27]),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int64'),
            ]


    class TestPrimitiveOp_13b970782982be40955767bdf9430825(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96aaacd9d57a811336f6298d67d831c7
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
            ]


    class TestPrimitiveOp_6c823d3a1bfd53d0c9dc124c80e58983(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a55308501b8885ffd76a09623ec892e
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[100, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_127ed340fc54688ca82da31b21befc2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96aaacd9d57a811336f6298d67d831c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            ]


    
    class PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_d93d0534b0b942ecd1d08654c3c81c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f0c9cb5b7db17fddcef2eb2c0b7807
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_f29ad336606768b6ba955dd50da68952(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d40d7f84f6af5fd0f502190fb463351f
        def get_inputs(self):
            return [
                paddle.uniform([220968], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_79290951a6ff4c3fc3cb572547116815(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd8b90c7bb88e33a8744a034b1b104be
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ca1d9eeb23fc51267ca0fe5208bd7f54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [5], [2], [2], [8]], dtype='int64').reshape([5, 1]),
            ]


    class TestPrimitiveOp_ca1d9eeb23fc51267ca0fe5208bd7f54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [5], [2], [2], [8]], dtype='int64').reshape([5, 1]),
            ]


    
    class PrimitiveOp_1a578264b00159ab9ece2d0cf2623909(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_6c52990a66f18815f2f7f480696f65e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a578264b00159ab9ece2d0cf2623909
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    
    class PrimitiveOp_724001a13e5f0e73a7caab72b47f7639(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b51bedac564f738d1e0687c3ff38377d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_724001a13e5f0e73a7caab72b47f7639
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 2, 2, 1, 0, 0, 1, 2, 0, 1, 1, 0, 1, 1, 1, 0], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_06781474cf46d8babcb252013673cfe1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_724001a13e5f0e73a7caab72b47f7639
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2, 1, 2, 2, 0, 1, 2, 1, 0, 1, 2, 0, 1, 1, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_f79f0fb4992da4307c6bdb7b512d68ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_724001a13e5f0e73a7caab72b47f7639
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 2, 1, 0, 1, 2, 1, 0, 1, 1, 2, 0, 1, 1, 2, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_1a5684b32640a42030e728ed89d31163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_724001a13e5f0e73a7caab72b47f7639
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 2, 1, 1, 2, 2, 1, 2, 0, 2, 2, 2, 1, 1, 0, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_4f63255f772a68865fc4c0cdeb477547(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_724001a13e5f0e73a7caab72b47f7639
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 2, 0, 0, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 0], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_a38598d9f9156bb94621fee01b864c3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_724001a13e5f0e73a7caab72b47f7639
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 2, 2, 1, 2, 0, 0, 0, 1, 1, 2, 1, 0, 2, 1, 0], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_2211973f850a94c8bb316a36c3c0932c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_724001a13e5f0e73a7caab72b47f7639
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 0, 1, 2, 2, 1, 2, 1, 0, 0, 2, 0, 2, 0, 1, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_81a028ec4df3916cc4bc8529ba3b0d5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_724001a13e5f0e73a7caab72b47f7639
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 2, 0, 1, 0, 2, 2, 0, 1, 2, 1, 1, 1, 1, 2, 0], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_7c9435b5b4c976a64bab02c661340525(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_724001a13e5f0e73a7caab72b47f7639
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2, 2, 1, 2, 0, 1, 0, 0, 1, 2, 2, 0, 0, 1, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_8d9582f2d6b4cc39f67343dc14f9d43b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_724001a13e5f0e73a7caab72b47f7639
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1, 1, 2, 0, 2, 2, 0, 0, 0, 2, 2, 1, 0, 0, 1], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_37d3da2a22d6c405a03b031dbf4173cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_724001a13e5f0e73a7caab72b47f7639
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 2, 1, 2, 2, 0, 1, 1, 0, 0, 0, 0, 2, 1, 1, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_cb2abef2effee4c055a82b13db55301b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_724001a13e5f0e73a7caab72b47f7639
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 0, 1, 1, 0, 1, 0, 1, 2, 0, 2, 1, 2, 0, 1, 1], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_c430d1f7c1fb295992a86cc3602cfc79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_724001a13e5f0e73a7caab72b47f7639
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 2, 2, 0, 2, 2, 0, 2, 1, 0, 1, 0, 0, 0, 0, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_e413ba4dfc5e598c2c4ea439b161026e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_724001a13e5f0e73a7caab72b47f7639
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 2, 0, 1, 0, 1, 2, 2, 1, 1, 1, 0, 0, 0, 0, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_0b4c716626f95767ea4c06a93d15ce52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_724001a13e5f0e73a7caab72b47f7639
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_3d07fdf39b2344943ba68ca707321a55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_724001a13e5f0e73a7caab72b47f7639
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 0, 1, 0, 0, 0, 2, 2, 1, 2, 0, 2, 1, 2, 0, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_2248fd0d68ffb8b87d23483545e11e7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d40d7f84f6af5fd0f502190fb463351f
        def get_inputs(self):
            return [
                paddle.uniform([185658], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_493d3bde75ccad833bed30c6b5c48eca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd8b90c7bb88e33a8744a034b1b104be
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_b6b462b8c0ba989a6b86a208d0bcc05a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [1], [6], [9], [2], [8], [2]], dtype='int64').reshape([7, 1]),
            ]


    class TestPrimitiveOp_b6b462b8c0ba989a6b86a208d0bcc05a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb198b575821f765d6d519647c28c8e
        def get_inputs(self):
            return [
                paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [1], [6], [9], [2], [8], [2]], dtype='int64').reshape([7, 1]),
            ]


    class TestPrimitiveOp_9baa9c37023c460a61908df5a4a09fb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a55308501b8885ffd76a09623ec892e
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            ]


    class TestPrimitiveOp_05059477b9896b10e517b4b72cbf1470(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96aaacd9d57a811336f6298d67d831c7
        def get_inputs(self):
            return [
                paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5]], dtype='int32').reshape([6, 1]),
            ]


    
    class PrimitiveOp_84cef30f4e9a05761db341dbabe14478(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_531245a777ba52aeffb71779f1b0fa2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[300, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_babbe9a3eb4f0120e4b4e58b7a54ad42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
            ]


    class TestPrimitiveOp_9379449d3e6f05be383bee75bf2d6058(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
            ]


    class TestPrimitiveOp_ca1d2c8050799bc296e34c0456570499(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[100, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_e91ee1004622206504fe2c744d65a4e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a31124b695ffeec8cddd1914e89a779
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.randint(low=0, high=3, shape=[2100], dtype='int64'),
            ]


    
    class PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d00ff82decacc7a75093eb57e011f714(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.12885119020938873, 0.11754139512777328, 0.37269124388694763, 0.08517523109912872]], dtype='float32').reshape([1, 4]),
                paddle.randint(low=0, high=3, shape=[2100], dtype='int64'),
            ]


    class TestPrimitiveOp_9379449d3e6f05be383bee75bf2d6058(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
            ]


    
    class PrimitiveOp_1a90be67dbd8625aee19ce6b13cb274c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a7bfb290075fc1cf3593461b70a203f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a90be67dbd8625aee19ce6b13cb274c
        def get_inputs(self):
            return [
                paddle.uniform([185691], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_398023f24ce303457e034df93b2a76fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f7fdf43fdc4c8f878079e91f04aef28b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_398023f24ce303457e034df93b2a76fc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = 0
            return paddle._C_ops.gather(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d4e8ddeaee2bda308350f432f524473b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4], [2]], dtype='int64').reshape([8, 1]),
            ]


    class TestPrimitiveOp_d4e8ddeaee2bda308350f432f524473b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4], [2]], dtype='int64').reshape([8, 1]),
            ]


    class TestPrimitiveOp_9379449d3e6f05be383bee75bf2d6058(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
            ]


    class TestPrimitiveOp_babbe9a3eb4f0120e4b4e58b7a54ad42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
            ]


    class TestPrimitiveOp_eb237b1677c242c91f26c0aed0629af6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a31124b695ffeec8cddd1914e89a779
        def get_inputs(self):
            return [
                paddle.to_tensor([9, 5], dtype='int32').reshape([2]),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int64'),
            ]


    class TestPrimitiveOp_5fbdd3163c618d371bda1e6f6f7f6b4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a31124b695ffeec8cddd1914e89a779
        def get_inputs(self):
            return [
                paddle.to_tensor([6, 0, 2, 8, 9, 6, 2, 5, 4, 0, 2, 4, 2, 2, 3, 5, 2, 4, 4, 1, 0], dtype='int32').reshape([21]),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int64'),
            ]


    class TestPrimitiveOp_43b4ceb7313d9db2e9e36e414b481476(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a90be67dbd8625aee19ce6b13cb274c
        def get_inputs(self):
            return [
                paddle.uniform([242991], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_84f4ef5e07d6815f6ea531bb1d1cde28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_398023f24ce303457e034df93b2a76fc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_ceaf87edcce4470829ccb9752754c3b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [0], [6], [1], [5]], dtype='int64').reshape([5, 1]),
            ]


    class TestPrimitiveOp_ceaf87edcce4470829ccb9752754c3b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [0], [6], [1], [5]], dtype='int64').reshape([5, 1]),
            ]


    class TestPrimitiveOp_1bf02200be374afc5169ba74b4130187(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            ]


    class TestPrimitiveOp_8c3a18c35e61bb11453f4388a6c5ed06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a31124b695ffeec8cddd1914e89a779
        def get_inputs(self):
            return [
                paddle.to_tensor([8, 5], dtype='int32').reshape([2]),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int64'),
            ]


    class TestPrimitiveOp_dd82a4a6688c29592312fecb53536b4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a90be67dbd8625aee19ce6b13cb274c
        def get_inputs(self):
            return [
                paddle.uniform([171888], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d6393d6600eacb0f4e120ae5862ecb60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_398023f24ce303457e034df93b2a76fc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_5e55b20cdf52b736903cded8f3ff29db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1]], dtype='int64').reshape([5, 1]),
            ]


    class TestPrimitiveOp_5e55b20cdf52b736903cded8f3ff29db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1]], dtype='int64').reshape([5, 1]),
            ]


    class TestPrimitiveOp_e3d2d89fc62e8dcedddf60cc932f7d6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5]], dtype='int32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_dd82a4a6688c29592312fecb53536b4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a90be67dbd8625aee19ce6b13cb274c
        def get_inputs(self):
            return [
                paddle.uniform([171888], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_d6393d6600eacb0f4e120ae5862ecb60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_398023f24ce303457e034df93b2a76fc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_bb30dc741f0c5cb8403238f27f6dc538(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1], [3], [3]], dtype='int64').reshape([7, 1]),
            ]


    class TestPrimitiveOp_bb30dc741f0c5cb8403238f27f6dc538(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [4], [1], [4], [1], [3], [3]], dtype='int64').reshape([7, 1]),
            ]


    class TestPrimitiveOp_01ae0cbed9f43a89b1eba3e5f0b163e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([3, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2]], dtype='int32').reshape([3, 1]),
            ]


    class TestPrimitiveOp_2677cbb035616dc2407925dea33815a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a90be67dbd8625aee19ce6b13cb274c
        def get_inputs(self):
            return [
                paddle.uniform([217413], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_cb26b1612ecc66163371ebb2b8c86bda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_398023f24ce303457e034df93b2a76fc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_b1fe40143bec4d947c370c3ff1d2f696(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[103, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_b1fe40143bec4d947c370c3ff1d2f696(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[103, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_9379449d3e6f05be383bee75bf2d6058(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1]], dtype='int32').reshape([2, 1]),
            ]


    class TestPrimitiveOp_9b20280183ed0a0655fb05e1593a28ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_34135e915f8b0f4bda5012b17fe22b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_d54d1f98233b81dc6b540f9f1d402f8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a31124b695ffeec8cddd1914e89a779
        def get_inputs(self):
            return [
                paddle.to_tensor([6, 6], dtype='int32').reshape([2]),
                paddle.randint(low=0, high=3, shape=[3549], dtype='int64'),
            ]


    class TestPrimitiveOp_0247379ae784060cb2340c311e050424(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4455467462539673, 0.04675282537937164, 0.2217721790075302, 0.248161181807518], [0.4585568606853485, 0.37466222047805786, 0.08961380273103714, 0.2054985761642456]], dtype='float32').reshape([2, 4]),
                paddle.randint(low=0, high=3, shape=[3549], dtype='int64'),
            ]


    class TestPrimitiveOp_aaff8c67f6663f16bfe5a51d12638f61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([7, 64, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            ]


    class TestPrimitiveOp_9b20280183ed0a0655fb05e1593a28ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_d60508f4e5d9a6d8ea14cbb9b48c0222(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a90be67dbd8625aee19ce6b13cb274c
        def get_inputs(self):
            return [
                paddle.uniform([86970], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_44b74100b29f2be35f26d737926b04ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_398023f24ce303457e034df93b2a76fc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_69de24b86d970ea09674de536792db64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [5], [1], [0], [0], [1]], dtype='int64').reshape([6, 1]),
            ]


    class TestPrimitiveOp_69de24b86d970ea09674de536792db64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [5], [1], [0], [0], [1]], dtype='int64').reshape([6, 1]),
            ]


    class TestPrimitiveOp_3b4f1a37b7bf1199b2520c58093556b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a90be67dbd8625aee19ce6b13cb274c
        def get_inputs(self):
            return [
                paddle.uniform([205923], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_7383c07dbe43550938f547d5ba913da7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_398023f24ce303457e034df93b2a76fc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_23dbdbc6846149e297c8387a2203e4f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [0], [8], [4], [1]], dtype='int64').reshape([5, 1]),
            ]


    class TestPrimitiveOp_23dbdbc6846149e297c8387a2203e4f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [0], [8], [4], [1]], dtype='int64').reshape([5, 1]),
            ]


    class TestPrimitiveOp_8ade664429c84aa9d697db1c4f6e2c9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a90be67dbd8625aee19ce6b13cb274c
        def get_inputs(self):
            return [
                paddle.uniform([153450], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_a999d0e2b53e332fc381578f4d32a208(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_398023f24ce303457e034df93b2a76fc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_76266dba1afffc0e125945f4ea505e42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1], [7], [4], [8], [3]], dtype='int64').reshape([10, 1]),
            ]


    class TestPrimitiveOp_76266dba1afffc0e125945f4ea505e42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1], [7], [4], [8], [3]], dtype='int64').reshape([10, 1]),
            ]


    class TestPrimitiveOp_2b611d0ee0599040f8603e0c4a1c3385(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4]], dtype='int32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_aee15e4d8a791f4c97834f31fea58e45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a31124b695ffeec8cddd1914e89a779
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.randint(low=0, high=3, shape=[4116], dtype='int64'),
            ]


    class TestPrimitiveOp_bd0d6ce9bbeb4fef5097e7f26a3891c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15030789375305176, 0.11389752477407455, 0.29615697264671326, 0.26190102100372314]], dtype='float32').reshape([1, 4]),
                paddle.randint(low=0, high=3, shape=[4116], dtype='int64'),
            ]


    class TestPrimitiveOp_1bf02200be374afc5169ba74b4130187(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            ]


    class TestPrimitiveOp_b44e03e05e89dc52b61e3da4cfef3d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a90be67dbd8625aee19ce6b13cb274c
        def get_inputs(self):
            return [
                paddle.uniform([113061], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_8fc1d5aa84e7f38bbd4f8a46a28854c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_398023f24ce303457e034df93b2a76fc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_88c63d044ef4784aaa54e3dad2e5aefb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[2], [6], [7], [8]], dtype='int64').reshape([4, 1]),
            ]


    class TestPrimitiveOp_88c63d044ef4784aaa54e3dad2e5aefb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[2], [6], [7], [8]], dtype='int64').reshape([4, 1]),
            ]


    class TestPrimitiveOp_1bf02200be374afc5169ba74b4130187(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            ]


    class TestPrimitiveOp_9b20280183ed0a0655fb05e1593a28ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_6f27e6adf6ae4ab560ab49d3f6073fb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a90be67dbd8625aee19ce6b13cb274c
        def get_inputs(self):
            return [
                paddle.uniform([123783], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_01a3a9010214322a421ae9238d1d3804(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_398023f24ce303457e034df93b2a76fc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_dead3fc3abede27ed18de878f3d73e19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[84, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_dead3fc3abede27ed18de878f3d73e19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[84, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_531245a777ba52aeffb71779f1b0fa2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[300, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_a7bfb290075fc1cf3593461b70a203f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a90be67dbd8625aee19ce6b13cb274c
        def get_inputs(self):
            return [
                paddle.uniform([185691], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_f7fdf43fdc4c8f878079e91f04aef28b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_398023f24ce303457e034df93b2a76fc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_a2338d76057ce5b03690a29f7f19d9d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4]], dtype='int64').reshape([7, 1]),
            ]


    class TestPrimitiveOp_a2338d76057ce5b03690a29f7f19d9d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1], [1], [5], [0], [9], [2], [4]], dtype='int64').reshape([7, 1]),
            ]


    class TestPrimitiveOp_2b611d0ee0599040f8603e0c4a1c3385(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4]], dtype='int32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_8ade664429c84aa9d697db1c4f6e2c9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a90be67dbd8625aee19ce6b13cb274c
        def get_inputs(self):
            return [
                paddle.uniform([153450], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_a999d0e2b53e332fc381578f4d32a208(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_398023f24ce303457e034df93b2a76fc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_62d069e9d67d3cf0071df125e4859bf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1]], dtype='int64').reshape([6, 1]),
            ]


    class TestPrimitiveOp_62d069e9d67d3cf0071df125e4859bf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[8], [4], [4], [2], [3], [1]], dtype='int64').reshape([6, 1]),
            ]


    class TestPrimitiveOp_96603a6badd444721c13fb34d25e320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_96603a6badd444721c13fb34d25e320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_96603a6badd444721c13fb34d25e320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_96603a6badd444721c13fb34d25e320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_96603a6badd444721c13fb34d25e320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_96603a6badd444721c13fb34d25e320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_96603a6badd444721c13fb34d25e320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_96603a6badd444721c13fb34d25e320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_96603a6badd444721c13fb34d25e320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_96603a6badd444721c13fb34d25e320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_96603a6badd444721c13fb34d25e320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_96603a6badd444721c13fb34d25e320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_96603a6badd444721c13fb34d25e320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_96603a6badd444721c13fb34d25e320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_96603a6badd444721c13fb34d25e320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_96603a6badd444721c13fb34d25e320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[49], dtype='int64'),
            ]


    class TestPrimitiveOp_369960a4fb68c4fe9211e667267c8f4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e98250a819fb0f02ba52e84ae9d7cf21
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_369960a4fb68c4fe9211e667267c8f4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e98250a819fb0f02ba52e84ae9d7cf21
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_8e316967898300aa503488f598d6d33b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e98250a819fb0f02ba52e84ae9d7cf21
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_8e316967898300aa503488f598d6d33b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e98250a819fb0f02ba52e84ae9d7cf21
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 9], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_c869328605148cf8f288af737add6c7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_9b20280183ed0a0655fb05e1593a28ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_28ebae9afd03c6401122cfad3e8afe8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a31124b695ffeec8cddd1914e89a779
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 2, 3, 5, 2, 4, 4, 1, 0, 6, 8, 6, 0, 6, 9, 3, 4, 9, 4, 0, 0, 7, 8, 6, 1, 9, 3], dtype='int32').reshape([27]),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int64'),
            ]


    class TestPrimitiveOp_cd7ef66e27a40dac9832277d1b45c579(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6], [7]], dtype='int32').reshape([8, 1]),
            ]


    class TestPrimitiveOp_ca1d2c8050799bc296e34c0456570499(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[100, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_9b20280183ed0a0655fb05e1593a28ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0]], dtype='int32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_bc5472f8bbb0cf87041909626199a46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_a7ccfbc876b0ab6cf4ddc6413336a0b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a90be67dbd8625aee19ce6b13cb274c
        def get_inputs(self):
            return [
                paddle.uniform([220968], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_cb214788efc9f58c88ed297af5ea9870(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_398023f24ce303457e034df93b2a76fc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_0fa4519f276f80b84596cca8e6c2e8d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [5], [2], [2], [8]], dtype='int64').reshape([5, 1]),
            ]


    class TestPrimitiveOp_0fa4519f276f80b84596cca8e6c2e8d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[6], [5], [2], [2], [8]], dtype='int64').reshape([5, 1]),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_1cbc65add68c5b9c29370bdae72a4db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[196], dtype='int64'),
            ]


    class TestPrimitiveOp_e63023d8fd976160b9b3de941285a0cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 2, 2, 1, 0, 0, 1, 2, 0, 1, 1, 0, 1, 1, 1, 0], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_7d2af5aeb23bb9733a11b89a97f8f2e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2, 1, 2, 2, 0, 1, 2, 1, 0, 1, 2, 0, 1, 1, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_4604e9d770ff03f6bd7e7deeaca92bb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 2, 1, 0, 1, 2, 1, 0, 1, 1, 2, 0, 1, 1, 2, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_750733e08b94e1494e5c00857dcda13f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 2, 1, 1, 2, 2, 1, 2, 0, 2, 2, 2, 1, 1, 0, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_4943f44ddaac5ea1a216a97120e8c555(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 2, 0, 0, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 0], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_ad3cdb046aaa92befbae25029b1ce864(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 2, 2, 1, 2, 0, 0, 0, 1, 1, 2, 1, 0, 2, 1, 0], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_47c89182d07d5ee7a7597011511fe677(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 0, 1, 2, 2, 1, 2, 1, 0, 0, 2, 0, 2, 0, 1, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_5a8e50bd0ca6993d9e522bcd2c600653(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 2, 0, 1, 0, 2, 2, 0, 1, 2, 1, 1, 1, 1, 2, 0], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_757ce4b0cc82f6080cc6d37f63cf3bfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2, 2, 1, 2, 0, 1, 0, 0, 1, 2, 2, 0, 0, 1, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_dd099384b0de49b1cf368910224e2b0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1, 1, 2, 0, 2, 2, 0, 0, 0, 2, 2, 1, 0, 0, 1], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_972eaf2fb94a593c2f3783e60e0a3358(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 2, 1, 2, 2, 0, 1, 1, 0, 0, 0, 0, 2, 1, 1, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_5adec63ee13ae75185a607c0af7a718c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 0, 1, 1, 0, 1, 0, 1, 2, 0, 2, 1, 2, 0, 1, 1], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_7d23b1ddf88e8a862d9f39462c50dbc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 2, 2, 0, 2, 2, 0, 2, 1, 0, 1, 0, 0, 0, 0, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_5ba262a094b89e2f7e5095c286c6ca0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 2, 0, 1, 0, 1, 2, 2, 1, 1, 1, 0, 0, 0, 0, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_e2acfe005acd9788294ff421e1b07bb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_429cc4c4c196bd57fa9b1246ea064bd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd68f4014cbaf26d08aea6775358f39
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 0, 1, 0, 0, 0, 2, 2, 1, 2, 0, 2, 1, 2, 0, 2], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_85c579607671c0b453c8223c9206d614(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a90be67dbd8625aee19ce6b13cb274c
        def get_inputs(self):
            return [
                paddle.uniform([185658], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_086fbd31125e2eb18fc1301dd45e0811(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_398023f24ce303457e034df93b2a76fc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[256, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_1e40d60165df2843511b57139999fc76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [1], [6], [9], [2], [8], [2]], dtype='int64').reshape([7, 1]),
            ]


    class TestPrimitiveOp_1e40d60165df2843511b57139999fc76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd59fc61753fd37d0aa27fcaa7e3e857
        def get_inputs(self):
            return [
                paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[9], [1], [6], [9], [2], [8], [2]], dtype='int64').reshape([7, 1]),
            ]


    class TestPrimitiveOp_1bf02200be374afc5169ba74b4130187(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5], [6]], dtype='int32').reshape([7, 1]),
            ]


    class TestPrimitiveOp_e3d2d89fc62e8dcedddf60cc932f7d6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84cef30f4e9a05761db341dbabe14478
        def get_inputs(self):
            return [
                paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0], [1], [2], [3], [4], [5]], dtype='int32').reshape([6, 1]),
            ]


    

if __name__ == '__main__':
    unittest.main()