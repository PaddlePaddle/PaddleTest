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
    class PrimitiveOp_866684f909cf8f3956b1e42bcb946f60(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_681faf274ccf1524f0e0d4814a77e243(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_083ec6ac45fd31c97b92bfc6d17c52d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_15d60192ec3650f2e5faf26e82d94dc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_083ec6ac45fd31c97b92bfc6d17c52d3
        def get_inputs(self):
            return [
                paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ec01645eb1a6590dd481c0d2bf6f93a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7d982c674fa3ee01aed891ea61deab33(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[3800, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd057f81cd23875b0e34db18cb03f468(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d982c674fa3ee01aed891ea61deab33
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f5a311be7378b40cf231288c318904a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_85262c91410f50bf3a612f95ed48e47e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[150, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b5a9563d671f7a2754f0ffd5d2a997a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85262c91410f50bf3a612f95ed48e47e
        def get_inputs(self):
            return [
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5b161872740ad0872122693a1a86b353(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[40, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bbde371de0ee8d10e62e0f2c6fd0cea6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b161872740ad0872122693a1a86b353
        def get_inputs(self):
            return [
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2dabe781f014ee0da83bed51d75684c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2a98139fe580aa0f4abc242610b3fae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c5fa79dfdc6d2a88e40537106549d4cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e2432e20c9a7cc01fb7633176dcb1bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fa79dfdc6d2a88e40537106549d4cc
        def get_inputs(self):
            return [
                paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31aa9b95b0aee72d2d34b9691f43e44d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cfa933106dd04cbf3943ddab776d567(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_17324e4555546db7c0d54df3c48e0ef9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[15200, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_53e72ac4ddc23a9951dc93ae62b2287d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17324e4555546db7c0d54df3c48e0ef9
        def get_inputs(self):
            return [
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f98dfb23dbbcafa0e7558ddca3f0a1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_87556725d83cf803e8a9abd4ff8b74f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[2204, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e15f8deaea61219a92207574d70731d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87556725d83cf803e8a9abd4ff8b74f5
        def get_inputs(self):
            return [
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f88af923d116a14771537c1300135d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[551, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab3ceaf00cd1f4694c298332ff9f6f2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f88af923d116a14771537c1300135d5
        def get_inputs(self):
            return [
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0067f56d319e8dd9d19ac58b201dc5af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[247, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa2e4f0fe4b23b5423973e411c1513b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0067f56d319e8dd9d19ac58b201dc5af
        def get_inputs(self):
            return [
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1112c2260776c83d44f56ff68fef1953(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[950, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41a78d50df06cd8692acbdf7920034c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1112c2260776c83d44f56ff68fef1953
        def get_inputs(self):
            return [
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3f48d79319e218dca56d4786b4f3158(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_361bb7945dec796dfca6ad14c9ee0fba(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[8816, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_10d93809c05da9640caed40dd5d6e511(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_361bb7945dec796dfca6ad14c9ee0fba
        def get_inputs(self):
            return [
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_76c61f96c29a74eed30d1a6879bdc04d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c58f5ff1a537150196683e010eef733c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76c61f96c29a74eed30d1a6879bdc04d
        def get_inputs(self):
            return [
                paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1e7c9fc10a3498e21fbb4c58b9c0243(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90a3855d5423c60d9a83059988193025(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ecccb1f37f6267fbcced6a9e4fdc482(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7bc42fbee98046300688262f91bcbf9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bd0c109f7377a8e15b59ec956289da8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1e560e4d5690ec54b0e43e53da69ebab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cbbb03ae06b47870dc0f24febc5ea437(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e560e4d5690ec54b0e43e53da69ebab
        def get_inputs(self):
            return [
                paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd057f81cd23875b0e34db18cb03f468(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d982c674fa3ee01aed891ea61deab33
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5ccd2a94e95c0076f86eaf3c2f870435(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[70, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_983ca3ab70d19a85aacde5b3666c4cc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ccd2a94e95c0076f86eaf3c2f870435
        def get_inputs(self):
            return [
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b9736394f73d7fac187f92b79a89ae7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_681faf274ccf1524f0e0d4814a77e243(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_19e61e22d67fee350b5a26a662946dee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fc6e8429731ccf486c9221bfb8ffe690(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19e61e22d67fee350b5a26a662946dee
        def get_inputs(self):
            return [
                paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ec01645eb1a6590dd481c0d2bf6f93a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b95a57c43407ac8e1a3d6df3788fa2f7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            return paddle._C_ops.sigmoid_cross_entropy_with_logits(input_0, input_1, None, False, -100)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c6a30e6f496cc99e65814dfa46234bef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b95a57c43407ac8e1a3d6df3788fa2f7
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f5a311be7378b40cf231288c318904a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18444cf1c7ad5f0fc74ac564d4409250(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b95a57c43407ac8e1a3d6df3788fa2f7
        def get_inputs(self):
            return [
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6e64cef179fbc5d52feac3cb9fdd2ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b95a57c43407ac8e1a3d6df3788fa2f7
        def get_inputs(self):
            return [
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2dabe781f014ee0da83bed51d75684c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2a98139fe580aa0f4abc242610b3fae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ebd90fd53126ab5c027046c5f8a5b8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19e61e22d67fee350b5a26a662946dee
        def get_inputs(self):
            return [
                paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31aa9b95b0aee72d2d34b9691f43e44d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cfa933106dd04cbf3943ddab776d567(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2adc4b0dc89dec73ce184035c7682a90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b95a57c43407ac8e1a3d6df3788fa2f7
        def get_inputs(self):
            return [
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f98dfb23dbbcafa0e7558ddca3f0a1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a92588df7963c5e95788e884d8f5bd43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b95a57c43407ac8e1a3d6df3788fa2f7
        def get_inputs(self):
            return [
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e91d835301457a9be4871384c31e00f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b95a57c43407ac8e1a3d6df3788fa2f7
        def get_inputs(self):
            return [
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2637b09cb5261527d642c13baae5f5c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b95a57c43407ac8e1a3d6df3788fa2f7
        def get_inputs(self):
            return [
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdcac5055d18ff0b92a36abde6736e22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b95a57c43407ac8e1a3d6df3788fa2f7
        def get_inputs(self):
            return [
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3f48d79319e218dca56d4786b4f3158(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ed8ef72fe9a205a25a84f71cfa423aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b95a57c43407ac8e1a3d6df3788fa2f7
        def get_inputs(self):
            return [
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_595048c3eb5377be7627801d7dcbf038(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19e61e22d67fee350b5a26a662946dee
        def get_inputs(self):
            return [
                paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1e7c9fc10a3498e21fbb4c58b9c0243(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90a3855d5423c60d9a83059988193025(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ecccb1f37f6267fbcced6a9e4fdc482(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7bc42fbee98046300688262f91bcbf9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bd0c109f7377a8e15b59ec956289da8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38a98cc399b7983b2dd9c24958b427fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19e61e22d67fee350b5a26a662946dee
        def get_inputs(self):
            return [
                paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6a30e6f496cc99e65814dfa46234bef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b95a57c43407ac8e1a3d6df3788fa2f7
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f8179773fab9d8ffa631622a7a478db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b95a57c43407ac8e1a3d6df3788fa2f7
        def get_inputs(self):
            return [
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b9736394f73d7fac187f92b79a89ae7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_866684f909cf8f3956b1e42bcb946f60
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b330a932bf902e494354ae7e42cca8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6e0b000acd2cca0dd2478cda5b8efe5
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()