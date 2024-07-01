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
    class PrimitiveOp_cedba2eb201a3db429333bd7ed9adbfc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_667bd71ba2d47b274927271c574f4ae1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cedba2eb201a3db429333bd7ed9adbfc
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_701c0b22b06166759f824c83013cceb2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = -1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 21504, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2374a9d1dbbe20af7edbe85b8b32d839(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_701c0b22b06166759f824c83013cceb2
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fcd19aec86b16df7ce095b66011c39ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9fd49641f7d01160b256d314c0d65ef2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd19aec86b16df7ce095b66011c39ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c99cc3ca3ea6882d97d75f6b8ec32072(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 24, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cc7a57d4c33ffe3978a38975df866748(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c99cc3ca3ea6882d97d75f6b8ec32072
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_52ffbb8ae4c8236f3a3188b26050dd41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 448, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8246e2d907d07186d6f3107f34b8108d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1a0a317bba84fac4a83156fb20458596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8246e2d907d07186d6f3107f34b8108d
        def get_inputs(self):
            return [
                paddle.uniform([129024, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([32256, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([8064, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2016, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([528, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b9605db5d34190328a91cbc7b542e2f7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f621e2052a28d3dac9a84578754c19f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9605db5d34190328a91cbc7b542e2f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 129024, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32256, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8064, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2016, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 528, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dc71b2b4ce912d6fc39c96fbb132300b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1, None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1, None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1, None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1, None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c9b0266cd00ad2e2a24d6b2ca2503755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc71b2b4ce912d6fc39c96fbb132300b
        def get_inputs(self):
            return [
                paddle.uniform([1, 129024, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32256, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8064, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2016, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 528, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6b7aba894b8ca294c46d17436c61afa6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    
    class PrimitiveOp_193e276d4dc53c991a59f28d40dfa631(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2c0d3e40a2043ddd0ceed0c1f80b5f0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_193e276d4dc53c991a59f28d40dfa631
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 152, 152], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 152, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1b6609bdd6a9299b0fd68f2a1b9494f4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_589ba8bc40ef1b6904939628f1078198(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b6609bdd6a9299b0fd68f2a1b9494f4
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_95654e1ecfb312059c706c08293f1b23(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5419e3bd63e4134da16e4da0bf177d19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95654e1ecfb312059c706c08293f1b23
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4568cc28a5070b7a144c20d81a279f30(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5361378694a31c6ad75ff8a4ada1f17d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4568cc28a5070b7a144c20d81a279f30
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4726118743419647, -0.5898312330245972, -0.09625720977783203, -0.42435988783836365, -0.17186817526817322, -0.5406213998794556], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.3495945632457733, -0.4441525936126709, -0.45715004205703735, -0.41611766815185547, -0.6559632420539856, -0.6102994680404663], dtype='float32').reshape([6]),
                paddle.to_tensor([0.5749874114990234, 0.8079026937484741, 1.0657973289489746, 0.878547191619873, 1.0153841972351074, 1.0460015535354614], dtype='float32').reshape([6]),
                paddle.to_tensor([1.1797178983688354, 0.674731969833374, 0.7607995867729187, 1.0119351148605347, 0.775779664516449, 0.7970899343490601], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_dfb7b72a58bf6dd1fb46eb6f58124bb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    
    class PrimitiveOp_b290a20b9f386391ddc6c70f35869902(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e5bdd8b60be6c0a43c6283b7d9dd1a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b290a20b9f386391ddc6c70f35869902
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    
    class PrimitiveOp_27a9e54228b95eaf48d93ede63917b56(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = -1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8471b81464687f3d766fe94b39f2efc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27a9e54228b95eaf48d93ede63917b56
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62e8dfb179fc985eac5f799e9c5d7bcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8246e2d907d07186d6f3107f34b8108d
        def get_inputs(self):
            return [
                paddle.uniform([115200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([28800, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([7200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1800, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([450, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4bc259a70930c6a1b25bdf91d47da9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9605db5d34190328a91cbc7b542e2f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 115200, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 28800, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7200, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1800, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 450, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e204da8c43e525da4af46ea9fa2d506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc71b2b4ce912d6fc39c96fbb132300b
        def get_inputs(self):
            return [
                paddle.uniform([1, 115200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 28800, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1800, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 450, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3ffbe35993479fde103f94ac2c55b2e9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, None, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, None, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc19d05fb0577412f98d945ebabbb82d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ffbe35993479fde103f94ac2c55b2e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d2f4ab04a3e148501b4f83f9b6318128(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f36a79031d19f9ade22484618626010f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2f4ab04a3e148501b4f83f9b6318128
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3f738c1d8584e892956793930d78a2b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be08ff495a90b6c0d8b8d8aa53c983bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9cc50f5c1192de53492493051599d446(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    
    class PrimitiveOp_461d923546c9e1b9f6d41dea496e3e06(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_01c2a1a5304ce7b567c651310b6411eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461d923546c9e1b9f6d41dea496e3e06
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 104, 104], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 36, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0cbbb0d3529c8af89479d6c1c894b658(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 60, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 60, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_29286bed01148b05d5fc583e88b9197e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0cbbb0d3529c8af89479d6c1c894b658
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 60, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a53c6e88ae1886f78055aee4643423b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_96f604ab75d2693ab35a29c61860325a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a53c6e88ae1886f78055aee4643423b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_38080445be453a6914e8f74ade8c5daa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c4163cbee4349c5805c670b391579e50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38080445be453a6914e8f74ade8c5daa
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3be79e4956df282068948c105c4d56d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f4adfe2b1b53bec0dea93e4863c77f0e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 150, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_22e60a15bfd494e9baed8622e5c357fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4adfe2b1b53bec0dea93e4863c77f0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 150, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fdea871a60d6fba5929b463d6ea8ed83(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0):
            input_0 = [arg_0_0]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 500, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5eb45f6e9559f83f5a7315a90c0f8804(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdea871a60d6fba5929b463d6ea8ed83
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_ea87c44da5f8b90619ca4475f041774d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 2
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 500, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[None, None, 1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_802de5501aa339657dc493ecb92e5361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea87c44da5f8b90619ca4475f041774d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5032b62db8b5a266ece5b2d7abfc67ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 9216, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2304, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b6c94b883c336c03204c6489105f6a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 9216, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2304, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5b3d1876bef63976853e8297ade6981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 9216, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2304, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ea114ee4fde2028d6337193f65669333(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[9216, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[2304, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[576, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7be5529bc7625b5eb44d68ef8e995997(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea114ee4fde2028d6337193f65669333
        def get_inputs(self):
            return [
                paddle.uniform([9216, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([2304, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([576, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0870b1643b6d8700e61e3818f978590a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[9216, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2304, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[576, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d5824ea6fc209068d08026d32e729e97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0870b1643b6d8700e61e3818f978590a
        def get_inputs(self):
            return [
                paddle.uniform([9216, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2304, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([576, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_481c3e73ce94adfdb5d85fd7e9541050(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = -1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12096, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12096, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_38579cbe68dfae0743f289977c2838c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_481c3e73ce94adfdb5d85fd7e9541050
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7be5529bc7625b5eb44d68ef8e995997(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea114ee4fde2028d6337193f65669333
        def get_inputs(self):
            return [
                paddle.uniform([9216, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([2304, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([576, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2264301399ba23f3e3c3d6850b54a808(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_850e0685ada8e02d4538668543c91319(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c178d4c3fd160393d9c1f823beca52fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ba9103aca0c480b8f8734e8c071c9c3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c178d4c3fd160393d9c1f823beca52fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1180bbd616f42cb804a77d8ac40473f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e82f8b0dcbe1389173792105a097c3e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68ae845cac21b689c8dc866c35a02c94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    
    class PrimitiveOp_f8367bddd16447487644cf634ef318ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0):
            input_0 = [arg_0_0]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1eec25ec0cc270afa62f00ed11b1b15d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b777cea8a3d903a6af3695f077a709a1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b7ee5c9a23261d130500acf13345b079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b777cea8a3d903a6af3695f077a709a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3ca8bae301dd20da0873d2be36930d42(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_80b62b5d6082cd0400e067c0e0b0dac9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ca8bae301dd20da0873d2be36930d42
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 48, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 48, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c47325a89c1df5419e84c1de33c21731(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0):
            input_0 = [arg_0_0]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2144a2b2e156b8135874f5268ca80387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
            ]


    class TestPrimitiveOp_4dbb328150a24fa01dd3546ea0d70ffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b777cea8a3d903a6af3695f077a709a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f4ea6c06980764be9bd2c5ca50216c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c178d4c3fd160393d9c1f823beca52fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1995643b0a18ba182ab7c1e6f2e877c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c178d4c3fd160393d9c1f823beca52fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5eb45f6e9559f83f5a7315a90c0f8804(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdea871a60d6fba5929b463d6ea8ed83
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_802de5501aa339657dc493ecb92e5361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea87c44da5f8b90619ca4475f041774d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_986489aec5dfb9e9c9a3f302a51c9675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c178d4c3fd160393d9c1f823beca52fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab7d35bce4cad2104e770b8614d04c50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
            ]


    class TestPrimitiveOp_af0d78d91b848ec5e2396916bb53bd23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b6609bdd6a9299b0fd68f2a1b9494f4
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_476ea7c27b97cd23d0291a352d1e12de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d38b8f6fc55e46ffbe1b63e672e92a6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cedba2eb201a3db429333bd7ed9adbfc
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d5cc13bd393f5d13c2d09dc6760a02b0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 34, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_529e5cdf80902f895247918a5230a40b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5cc13bd393f5d13c2d09dc6760a02b0
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 34, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0350670b8826a8a7762a97f10e5e35c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f58b59d798d61de627c83e165bd25e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0350670b8826a8a7762a97f10e5e35c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 36, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 160, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e5bdd8b60be6c0a43c6283b7d9dd1a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b290a20b9f386391ddc6c70f35869902
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_798804ddfda289f5a3ee96c3865f6c2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cda5b9e05a58f268b3218ea23cc069f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
            ]


    
    class PrimitiveOp_7a7769b4a30275dd5f81e65dd2ddc612(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce4dc25cfa40497fef9cbe3d353f1367(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a7769b4a30275dd5f81e65dd2ddc612
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba8f85b7b4609aef3a7ba7a40af226fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dfb7b72a58bf6dd1fb46eb6f58124bb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_f8f4e7bdd013721ec011ccdae0855a07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27a9e54228b95eaf48d93ede63917b56
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a357c84dfbd5403362aec51a533f5e98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cedba2eb201a3db429333bd7ed9adbfc
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbeb25ed3ac15e2a7558ab1c6c2694a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7c7fd5ec4a261805c77dd979e1f8250(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd19aec86b16df7ce095b66011c39ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b9cc6d66add140a4ed757605bd424286(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 25, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7520db0b266e83ff5918caf46ba0615d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9cc6d66add140a4ed757605bd424286
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e23d286bb5f7e9bedde8d3d2def1658e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27a9e54228b95eaf48d93ede63917b56
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61396c6cdcd467e3895fdb800ef8a114(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
            ]


    class TestPrimitiveOp_57a4392415c235c02b2c53143177f7e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c8b0122089dd40d331983838239379ed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c041aa06005eb4122e8d253d206fd276(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8b0122089dd40d331983838239379ed
        def get_inputs(self):
            return [
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4aded8307ddca406fe4821e8c0273fd6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aafbfd079c649f1ad09d20c5add41fe9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4aded8307ddca406fe4821e8c0273fd6
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_29b9fa81183a10c24fc18b52c3142521(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0):
            input_0 = [arg_0_0]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_981733171efe7cdfeb59f4f4e9b3a07e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29b9fa81183a10c24fc18b52c3142521
        def get_inputs(self):
            return [
                paddle.to_tensor([[4], [7], [8], [2]], dtype='int64').reshape([4, 1]),
            ]


    class TestPrimitiveOp_e4bf4329285749c98b247b3989d7b2eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_00a6fffbd2de9c190c75be6d551b8399(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([9, 9, 3, 6], dtype='int32').reshape([4]),
            ]


    
    class PrimitiveOp_05fa05daa6c838ed81a83ab37a187aed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0):
            input_0 = [arg_0_0]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ad9034741bd1b9f102d001ad5a57b869(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05fa05daa6c838ed81a83ab37a187aed
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4, 28, 28], dtype='int32'),
            ]


    
    class PrimitiveOp_be9eb669746fd9491332d4aaee25f5ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0):
            input_0 = [arg_0_0]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f95b1c4b242e44d30abfd224c19819b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be9eb669746fd9491332d4aaee25f5ee
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2847067415714264, 0.4610000550746918, 0.04074212908744812, 0.044464848935604095], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_74cf2e2cc3d006ef395ac70fb711652e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cedba2eb201a3db429333bd7ed9adbfc
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dcc3c53b507b87f2f9d2bfe076953e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0cbbb0d3529c8af89479d6c1c894b658
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 60, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_426514ca536d2838ba41f576fbc761de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29b9fa81183a10c24fc18b52c3142521
        def get_inputs(self):
            return [
                paddle.to_tensor([[4], [7], [8], [2], [2], [9]], dtype='int64').reshape([6, 1]),
            ]


    class TestPrimitiveOp_632a85866daac99cf6111d95750a9188(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([9], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e5fd1908a1cc3d4eb4f76e3a4b964b79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([3, 6, 3, 3, 1, 7], dtype='int32').reshape([6]),
            ]


    class TestPrimitiveOp_d8e36e5f8dd3adf06fa3925dfe008f7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05fa05daa6c838ed81a83ab37a187aed
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[6, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_cb09cb162a45d374ecca4f6d5df1397d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be9eb669746fd9491332d4aaee25f5ee
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3677123486995697, 0.03161490708589554, 0.4054182469844818, 0.1435633897781372, 0.4872145354747772, 0.4321914613246918], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_1d32ec4c05cf76f3fbb3460efddfc3a8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd4b186e231126ab683ccb71e4b196c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d32ec4c05cf76f3fbb3460efddfc3a8
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 19, 256, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_72ea7f5380e3e28844c8115cb08e6cc8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb136edc643a181f21a3fa53011b043d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72ea7f5380e3e28844c8115cb08e6cc8
        def get_inputs(self):
            return [
                paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_667c2516da755eebfa55d0125d7dbe78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
            ]


    
    class PrimitiveOp_f0a5045cb4a40923ed322b0780394b65(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = -1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a0c498776046c4a4351f5cfeaad4cffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40ca4b468af1456038952eae31d54ddb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27a9e54228b95eaf48d93ede63917b56
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a215e848eb49f1bf20e6de615ae10165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd19aec86b16df7ce095b66011c39ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4776ad40c7d611677159fa44bd922cdf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 20, 30], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b1e4ac274c28a6ab0c14d6851378178a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4776ad40c7d611677159fa44bd922cdf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f54fceef130d50e8e6789b59cf136331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe7f258cab02450e3ffb4225afa91410(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_331e80718f2476d55264aacbd707b2d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce434da45bb5b39d0df9b5f4f1c57fb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1024, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_52fc7ea0f365f29676573ef33a88d3a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1024, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cb7254f5ace20d05471ae6649565f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c3ecbf932be44b97eea14835a77278c6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4096, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1024, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb7c434268f2f4b9fdcd23f318b42ac5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3ecbf932be44b97eea14835a77278c6
        def get_inputs(self):
            return [
                paddle.uniform([4096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_256106162ec7c45cbbfa916dd13b672c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4096, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1024, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_99e6021d2d1c3d8a1cd8ae511eba7a73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256106162ec7c45cbbfa916dd13b672c
        def get_inputs(self):
            return [
                paddle.uniform([4096, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f918e444273e290bc7d24fc71df41c1e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = -1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 5376, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 5376, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e72a8922f3e29b396b86524b4de469da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f918e444273e290bc7d24fc71df41c1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb7c434268f2f4b9fdcd23f318b42ac5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3ecbf932be44b97eea14835a77278c6
        def get_inputs(self):
            return [
                paddle.uniform([4096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67492b5f94de583300cb8f548db3c33b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    
    class PrimitiveOp_17165537f396430d39f8755f4216b690(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6aa25ffd78df3153c30c04b5dce25806(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17165537f396430d39f8755f4216b690
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_849ec0c0aebc15b4e68b00d7bfade008(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a173a0368969c28e1c921696193878f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3fa590ed4b72d434fdbbcc1fc8f86e73(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_416bd1870883375b10e560fcc7ae199f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fa590ed4b72d434fdbbcc1fc8f86e73
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f948059304cf7ee07d286af6e68650c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8471b81464687f3d766fe94b39f2efc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27a9e54228b95eaf48d93ede63917b56
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6137a6e6f59f9c3c5d40fbba9932637d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b777cea8a3d903a6af3695f077a709a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 96, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5663e52157ce5eccc8fdf4993cbebc49(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d00c59008e3eaf4c65eff720d4d9d1fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5663e52157ce5eccc8fdf4993cbebc49
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6eded2b9f0650d97fa6423c7db131455(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 224, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 128, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 128, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 128, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 128, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4beb79ea56326cbd79d002f14167e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
            ]


    class TestPrimitiveOp_a0565095165bef0c6fc137fd2d0eaf5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4568cc28a5070b7a144c20d81a279f30
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4631315767765045, -0.3884684443473816, -0.7395162582397461, -0.6278271675109863, -0.1999669075012207, -0.19013196229934692], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.5041857957839966, -0.5462598204612732, -0.20084482431411743, -0.3272498846054077, -0.5095967054367065, -0.5581246018409729], dtype='float32').reshape([6]),
                paddle.to_tensor([0.8244056701660156, 0.9471744894981384, 0.8348401784896851, 0.64537513256073, 0.8654975891113281, 1.1398656368255615], dtype='float32').reshape([6]),
                paddle.to_tensor([1.0452922582626343, 0.6076709628105164, 0.9019972681999207, 1.311782717704773, 0.950197696685791, 0.6846595406532288], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_7e9a6d59e6e3204394e72adda01ebb04(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d3b940171227a37c91c76188d55c4df0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e9a6d59e6e3204394e72adda01ebb04
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_76192ba53b8ae9a3003a875d1e816a69(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e639b0c65dba27db68009ad11adf45e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76192ba53b8ae9a3003a875d1e816a69
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3a875dd0bddb5393f734414cccf5ce9b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9bae527f6332e6a8ab36b183916a515(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a875dd0bddb5393f734414cccf5ce9b
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21fc90ffb38ee649496f0b8116755cd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95654e1ecfb312059c706c08293f1b23
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_576d0f4b0cc4c0987bf4031d8b112e0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96f7074ba1d88e0ae8be6dc870a790e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ef832da62bf203d07917fb50a433ab84(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b3742fa4bab82599fa10911274c8241(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef832da62bf203d07917fb50a433ab84
        def get_inputs(self):
            return [
                paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            ]


    class TestPrimitiveOp_6220f5ac373fe3d458f3fe36d6a0191b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e9a6d59e6e3204394e72adda01ebb04
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_98732c43e03d7e2295c71d28eccbb1cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 512, 97, 97], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1478ca5d7c7efa31472dfb506d40d7dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98732c43e03d7e2295c71d28eccbb1cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a173a0368969c28e1c921696193878f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1fda05c52e1cbb18aaa78847c335cbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b16035529c8332e7f1fd5f8a726561bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b777cea8a3d903a6af3695f077a709a1
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 48, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f79cb3eca39f5ecb7cdb672f686601d4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3c832398a75ae36c54104ccd5d294e39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f79cb3eca39f5ecb7cdb672f686601d4
        def get_inputs(self):
            return [
                paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_789ae805092a9c3036df584d795fc813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4aded8307ddca406fe4821e8c0273fd6
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_789ae805092a9c3036df584d795fc813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4aded8307ddca406fe4821e8c0273fd6
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cf6a436ca86e7216168f4d3af16cd652(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8ae76fa3938bb5c8dd45cde9095da20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf6a436ca86e7216168f4d3af16cd652
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f83ab41ccd63166f0a43582cb5eaa6cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf6a436ca86e7216168f4d3af16cd652
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c663cfe3f9b92bd38862448130e415e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c663cfe3f9b92bd38862448130e415e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_42b1750dc4443c8c7dadf5813b1237c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e5480be701259c01c0cb319cf7ba0584(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42b1750dc4443c8c7dadf5813b1237c7
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03482a1ad1d861a05d833aa24dad1e48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42b1750dc4443c8c7dadf5813b1237c7
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cf98e54de563340e5f1354c88c94c76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7
        def get_inputs(self):
            return [
                paddle.uniform([3, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_8dfcb2e6f1328e83d4665b11b1114ffc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17165537f396430d39f8755f4216b690
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2dfcab43f14183003d4dea2b01bbd1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_193e276d4dc53c991a59f28d40dfa631
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 160, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_83aed82573b587f40a8690d6278abe50(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8cdf9e27a6beeaf9443ba10a4072fe2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83aed82573b587f40a8690d6278abe50
        def get_inputs(self):
            return [
                paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99106bd02eb369ba723c45e56ef11785(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8246e2d907d07186d6f3107f34b8108d
        def get_inputs(self):
            return [
                paddle.uniform([139392, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([34848, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([8712, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2178, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([528, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9c03b6023c28bce13c6fdf700db2bea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9605db5d34190328a91cbc7b542e2f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 139392, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 34848, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8712, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2178, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 528, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bb9b44b688bc30c33dd66514fbc7f77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc71b2b4ce912d6fc39c96fbb132300b
        def get_inputs(self):
            return [
                paddle.uniform([1, 139392, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 34848, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8712, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2178, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 528, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9fdc7db94861d01511df1efce4d1fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b777cea8a3d903a6af3695f077a709a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccc68a0d604876953e2fdcbb9ff9cc81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c178d4c3fd160393d9c1f823beca52fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_59c83a5ec9bb2b16eacd992488268797(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_96d04f6371242cd0dddc9b22a392dbb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59c83a5ec9bb2b16eacd992488268797
        def get_inputs(self):
            return [
                paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8638b9ffdfb34bb136bf8d68d607cd79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 512, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9125ac8796a51f0a3e48be2480bba507(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0cbbb0d3529c8af89479d6c1c894b658
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 168, 168], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 60, 168, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3cf1c94b3a0881eac38a308b04db8bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_374727ee74c121edc1c7e9134332071f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 6], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_d65a0bac842ddee96dcc7779e44f64a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c178d4c3fd160393d9c1f823beca52fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_837e96852dfc9b1502664d9f5ef83d65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c178d4c3fd160393d9c1f823beca52fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e143dc35a8919b3d6fe43bd3448fcc89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4aded8307ddca406fe4821e8c0273fd6
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e143dc35a8919b3d6fe43bd3448fcc89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4aded8307ddca406fe4821e8c0273fd6
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3e0a46353a8c176f8911f632969766e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf6a436ca86e7216168f4d3af16cd652
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ede9b5d64619c282519c1552805c31e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf6a436ca86e7216168f4d3af16cd652
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_566563291aa2e4db24a206473a1cb374(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_566563291aa2e4db24a206473a1cb374(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_750a191a32abdd2b22c69993b3e0bb79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42b1750dc4443c8c7dadf5813b1237c7
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d023d2431e104c18382e647119f68a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42b1750dc4443c8c7dadf5813b1237c7
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_778b636d2c305bbf658e842656c69f4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72ea7f5380e3e28844c8115cb08e6cc8
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d2b023da1a9456d668deee2d9825f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_548407ce7dc0862e15e941f603ae96c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5cc13bd393f5d13c2d09dc6760a02b0
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 160, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 34, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_569af70052f37f79920fa1f142e3e387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4aded8307ddca406fe4821e8c0273fd6
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_785362186c98070dc9d40614b070c052(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9d1ec0e13e346cf29abc81881f31ae22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_785362186c98070dc9d40614b070c052
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0c498776046c4a4351f5cfeaad4cffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64259c1a3bc05e5e69c8d543e00a33db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 448, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a73d0ccd91bfacb12c307f63ed5d228e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 28, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 36, 28, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e5bdd8b60be6c0a43c6283b7d9dd1a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b290a20b9f386391ddc6c70f35869902
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_454a66eac92b62e00c72e9e3d7ef339c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b777cea8a3d903a6af3695f077a709a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 76, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8f8e5fb1fb4c3841b6ba3d0699a35ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27a9e54228b95eaf48d93ede63917b56
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6dac4538f3e283436077c377d02836ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef832da62bf203d07917fb50a433ab84
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            ]


    class TestPrimitiveOp_68a0b42cf2235a6df0b3babf8ed75d67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0fe198348526e8d1727bfa721d82669(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([6, 9], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_5fbade50b83f90e80b2694c42c92e5ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95654e1ecfb312059c706c08293f1b23
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c39b6475e5ca9adcfbff590dc5fd9ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 6400, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1600, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 400, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5736a577117faafb8551cb67a367ea67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 6400, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1600, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 400, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d8443db498b0d616a808784bdedcb2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 6400, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1600, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 400, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8a2e6ff787253e441b195540743228cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6400, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1600, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[400, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_278d7b5916dc633fb62b12117e2654ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a2e6ff787253e441b195540743228cd
        def get_inputs(self):
            return [
                paddle.uniform([6400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1600, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([400, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c3565c436b52cfda4b0ddcf9f4b27aeb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6400, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1600, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[400, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b1cf74b4cee24c04a7ea44b6b78e864(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3565c436b52cfda4b0ddcf9f4b27aeb
        def get_inputs(self):
            return [
                paddle.uniform([6400, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1600, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([400, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6bfcbd8bf3467cc4be5d4f30d36248d8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = -1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8400, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 8400, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ac6a54a569dea8b8d953d2976339edc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bfcbd8bf3467cc4be5d4f30d36248d8
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_278d7b5916dc633fb62b12117e2654ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a2e6ff787253e441b195540743228cd
        def get_inputs(self):
            return [
                paddle.uniform([6400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1600, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([400, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f8e461697cea38a09f6e694aab4748dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[49, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9353bb18c9b0f8a3b6e8ab4b311f9fd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8e461697cea38a09f6e694aab4748dd
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_79de388ccfd363dffdd5da0172cce877(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_61deb026ee026414b41f50eaa4923a5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79de388ccfd363dffdd5da0172cce877
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ea1b98075328286a1bc06c72a75a3c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_802caa5c66bca80180e24fcd38407a72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
            ]


    class TestPrimitiveOp_bc19d05fb0577412f98d945ebabbb82d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ffbe35993479fde103f94ac2c55b2e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f36a79031d19f9ade22484618626010f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2f4ab04a3e148501b4f83f9b6318128
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36af32df93a09e70d537287e2e586019(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0350670b8826a8a7762a97f10e5e35c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 36, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c5f9b8ac1d35b15b1e5d19a8d67c151(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27a9e54228b95eaf48d93ede63917b56
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f579ffdd1f269b7810026e3cf3962e14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc14f20472679404b3a54230c38b24f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72ea7f5380e3e28844c8115cb08e6cc8
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1d203906049cdc63985b4d54776d3a2d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2a58f7e11c2cc4a0882bc21f8ff1686f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d203906049cdc63985b4d54776d3a2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bfe813ac1f5fddf47f64e743710fec7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7
        def get_inputs(self):
            return [
                paddle.uniform([7, 64, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 64, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 64, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 64, 7, 7]),
            ]


    class TestPrimitiveOp_f3829a9feefb7e1be0f262610c74af1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_193e276d4dc53c991a59f28d40dfa631
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 136, 136], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 136, 136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c36b132d6f82af0f310660ec65006d34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6dac4538f3e283436077c377d02836ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef832da62bf203d07917fb50a433ab84
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            ]


    
    class PrimitiveOp_9d82ab4db138d32ce55e201e25646f24(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 2
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 4, 13, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4, 1, 13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_48963f7e7d4a67d5d9c4cd0310f83752(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d82ab4db138d32ce55e201e25646f24
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 13, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4, 1, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3a8475c81c91df83d65c22cc6bff777e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_91862feb95a176b3344ecea14a9410e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a8475c81c91df83d65c22cc6bff777e
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_dcf49992b10652342be9e82b35b033a0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='int32'),
                paddle.static.InputSpec(shape=[2], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_193910cee2ba8bbf232b0ae161db6974(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcf49992b10652342be9e82b35b033a0
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 2, 3, 4], dtype='int32').reshape([4]),
                paddle.to_tensor([0, 0], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_d023b2dac6266d44330d7a40a1a5c59b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_608a6cdc14ce69cecbc07d55f31c365c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_09298f9072dd4613ce71747c7945d79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_608a6cdc14ce69cecbc07d55f31c365c
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21366fcf6be52b651e86e542c3fb2520(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b6609bdd6a9299b0fd68f2a1b9494f4
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4242d4da9d5d50a03cceafda713e7e15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8246e2d907d07186d6f3107f34b8108d
        def get_inputs(self):
            return [
                paddle.uniform([182400, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([45600, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([11400, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2850, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([741, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfa2e7de4bf782506cb4ce1120374f2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9605db5d34190328a91cbc7b542e2f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 182400, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 45600, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11400, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2850, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 741, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53337e81e52e3c1ce64b802ba6897fad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc71b2b4ce912d6fc39c96fbb132300b
        def get_inputs(self):
            return [
                paddle.uniform([1, 182400, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 45600, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11400, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2850, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 741, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ea7e373d96a6bd2637a713058d896c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_861c1ea1a563a4232b2fad5de2ea6a06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0727feb16a2dcf76af9bbe17d31c51f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b777cea8a3d903a6af3695f077a709a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_637d4d4534aaedd7117317c9514f8f79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_515c0d132679c20ea18ffc505519b787(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
            ]


    class TestPrimitiveOp_bed0b408f2530799bbce2286bf92f110(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ea775da8c33ad52bd85703c9aa70aedc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1e096ef194f7705cbccaf8a9e0fa55f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea775da8c33ad52bd85703c9aa70aedc
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49734b8f5b16406c9f364ef4901979aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_193e276d4dc53c991a59f28d40dfa631
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 104, 104], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91d057b90962e94f3dd6a7ad14b77a45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b777cea8a3d903a6af3695f077a709a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 184, 184], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 184, 184], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2930dc8d8f3add326f21394f6d894dde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
            ]


    class TestPrimitiveOp_7715dd13280c0b6ad461f205f63cb74c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c178d4c3fd160393d9c1f823beca52fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fd9c4cfa40e4081fe8de663e74004689(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[784, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[3136, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1224d1e3fe67c34173107bf151285623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd9c4cfa40e4081fe8de663e74004689
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([784, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3136, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bfe26245ce7b8b223e33999d07467a06(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[196, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[784, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[3136, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5d0da5da0701ae7d76bebe74f4a85545(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfe26245ce7b8b223e33999d07467a06
        def get_inputs(self):
            return [
                paddle.uniform([196, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([784, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3136, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d62d0c16229991b389fdbee9d0970e9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[196, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[784, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3136, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff9e22101115632ad7629860568387d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d62d0c16229991b389fdbee9d0970e9e
        def get_inputs(self):
            return [
                paddle.uniform([196, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([784, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3136, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0981cb4f48b75a57e1f92e789bdbc72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c39ddce95c30234c45d3b4af3e7f8b6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4aded8307ddca406fe4821e8c0273fd6
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c39ddce95c30234c45d3b4af3e7f8b6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4aded8307ddca406fe4821e8c0273fd6
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fddadf7bae010111bdf51f19c62c3abc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf6a436ca86e7216168f4d3af16cd652
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e83e7ff0670036f91a65ad7536e70c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf6a436ca86e7216168f4d3af16cd652
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b075103dac7084a26e00e34515705f5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b075103dac7084a26e00e34515705f5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f875f4b10aeee4acc8363b9b5ec2317d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42b1750dc4443c8c7dadf5813b1237c7
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64ebc394c87605d2fc48d75a332f2539(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42b1750dc4443c8c7dadf5813b1237c7
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b17e72849199669778669a0de675595(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8246e2d907d07186d6f3107f34b8108d
        def get_inputs(self):
            return [
                paddle.uniform([65280, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([16320, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4080, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1020, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_edc14741561f96d9e34c3d9b00795f9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9605db5d34190328a91cbc7b542e2f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 65280, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16320, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4080, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1020, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_781f7fc6594c2292a2dccfeb7cea7f49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc71b2b4ce912d6fc39c96fbb132300b
        def get_inputs(self):
            return [
                paddle.uniform([1, 65280, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16320, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4080, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1020, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8aa0baec31fd4a9b706cb11eab8f58c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51f72c0f2c82fdf8ae9f5324c9f9d98e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7
        def get_inputs(self):
            return [
                paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    
    class PrimitiveOp_d408e5a64902e2c528e3bc4db950c9c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 200, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 200, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea1723ffa7f460a14f2eae678425488a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d408e5a64902e2c528e3bc4db950c9c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 200, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d44c59394b68087235efdd2ae7e543d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
            ]


    class TestPrimitiveOp_67492b5f94de583300cb8f548db3c33b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_8b261facfda4d39a8411366f5df5831f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b54beaad36878f6fbe33a902fdbc593e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7485630d536cea35d1561fdc4452c0c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17165537f396430d39f8755f4216b690
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67492b5f94de583300cb8f548db3c33b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    
    class PrimitiveOp_fc3d6ba772aba40377797242c2c054d4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_131c82038bb6f0355b489519f5976b52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc3d6ba772aba40377797242c2c054d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_066ded59fd3d558e40f24c13a55abc29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 512, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_703a1028a6969ba190f166ca310c655d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_acaf03e16abdb01a1086da742172f60b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b99096915a56c816194e84ab1bd5467b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_acaf03e16abdb01a1086da742172f60b
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ca3d8df20be6cd83b316cf8672505df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cedba2eb201a3db429333bd7ed9adbfc
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48bf28cc5cd7b0c59a7e0e0c58cc75d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c178d4c3fd160393d9c1f823beca52fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c82b18f178a91f93e78db8d49fad2fc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59c83a5ec9bb2b16eacd992488268797
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58800861b943bf090adbc3a7c4e11d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b777cea8a3d903a6af3695f077a709a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22a5753c61b4295f0f6bf67febeb96e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 232, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 232, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_072f08e60008f3d84dda8a6451fc7e94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6dac4538f3e283436077c377d02836ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef832da62bf203d07917fb50a433ab84
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            ]


    class TestPrimitiveOp_1571d4199b7cbfd9837b4d4959273a25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59c83a5ec9bb2b16eacd992488268797
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94a9274e7a48563752533534b4ecaa4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03678ed7a4238b62b25c44bdb2207ef6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d408e5a64902e2c528e3bc4db950c9c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 200, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b7aba894b8ca294c46d17436c61afa6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_5373e9e13a9525dbc4a44cf84606954c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc7b5a32b4f842bef550d252ae27b8a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc3d6ba772aba40377797242c2c054d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bc1c5133aef4aa416bdf1c890d6db288(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf7e57b0dd711100ca9c7c252f419a19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc1c5133aef4aa416bdf1c890d6db288
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1eec25ec0cc270afa62f00ed11b1b15d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ea811d854991b6b5cfa9ce910fac8c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1dde18f45a2a20f0f44bb42bf0ff413(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 14, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 14, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef4a12583e877773ffc6aacd5699cbeb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cedba2eb201a3db429333bd7ed9adbfc
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 42, 42], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de810cdf82f89ad59e8eaf9ef0a4ab29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_785362186c98070dc9d40614b070c052
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_541344029c7483ddb5db1e807490eb02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76192ba53b8ae9a3003a875d1e816a69
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c0320eced336688c7d6b1f5679a57d4d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b28547009742ca54ae8e70eaa9200174(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0320eced336688c7d6b1f5679a57d4d
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7de0deb7d07e689fb38a1055f71d2be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b6609bdd6a9299b0fd68f2a1b9494f4
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9eb83f7be654a4f4ca222ad59bc6bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b11311905511fba2d3192dc762a732dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 100, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 100, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_923d886e3c6bbf556ef2f1cbf79a284d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b11311905511fba2d3192dc762a732dd
        def get_inputs(self):
            return [
                paddle.uniform([22, 100, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 100, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51f72c0f2c82fdf8ae9f5324c9f9d98e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7
        def get_inputs(self):
            return [
                paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_5b12949bd43199e341a94782653e9c2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b26eff381010e9d3dbc0e33c70f339b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_df2ad9edf873f0a891184aae3978c449(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ed3838286288dfdefb306138f5c16ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df2ad9edf873f0a891184aae3978c449
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3166e46ed71a455b07a05e9d3d7f77d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29b9fa81183a10c24fc18b52c3142521
        def get_inputs(self):
            return [
                paddle.to_tensor([[4], [7], [8]], dtype='int64').reshape([3, 1]),
            ]


    class TestPrimitiveOp_e4bf4329285749c98b247b3989d7b2eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_efbbca4f20284a3b4996bcc2a704716c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 9, 9], dtype='int32').reshape([3]),
            ]


    class TestPrimitiveOp_659f43e482858296a61f52f77abc9894(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05fa05daa6c838ed81a83ab37a187aed
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_fce6a30e9e5e700a195534986806c162(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be9eb669746fd9491332d4aaee25f5ee
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06292838603258133, 0.30279308557510376, 0.39033249020576477], dtype='float32').reshape([3]),
            ]


    class TestPrimitiveOp_b4e6dfbb83c81de5e98e92132cfc98aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 512, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1ed2269484b12193c05a75314bfddfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc3d6ba772aba40377797242c2c054d4
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bed0b408f2530799bbce2286bf92f110(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ca9448712d56b223df445a9b76e80fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27a9e54228b95eaf48d93ede63917b56
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_da27c91c26d9ae4830ac8baee21ea01f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_38320120885bc12bacc625a162467e55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da27c91c26d9ae4830ac8baee21ea01f
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e0d0b3b18fbc97b54a3c0472ae1ca37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c178d4c3fd160393d9c1f823beca52fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91f3e8cfc3dc1f76c687591215954494(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.050307195633649826, 0.24958865344524384, 0.12819220125675201, 0.15536218881607056], [0.31391292810440063, 0.12240537256002426, 0.3002643585205078, 0.22792692482471466]], dtype='float32').reshape([2, 4]),
            ]


    
    class PrimitiveOp_41baac1cdf4b612b078c94271e932faf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0):
            input_0 = [arg_0_0]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_edbf8973e999477ff07c5dcd6193a3d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41baac1cdf4b612b078c94271e932faf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.37339529395103455, 0.30681005120277405, 0.14150168001651764, 0.0019667954184114933]], dtype='float32').reshape([1, 4]),
            ]


    class TestPrimitiveOp_912d1e70b9821b23eaf10bdcc0e9608b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08483082801103592, 0.046015478670597076, 0.15802784264087677, 0.42296135425567627], [0.37927430868148804, 0.43439051508903503, 0.4780726134777069, 0.20804838836193085]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_56a7bf7e135915458384964a02f52717(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b11311905511fba2d3192dc762a732dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e50ce9a3bb6f5c4f25fa24dfce50c59a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 4624, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1156, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 289, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e90ae374b3d80d508065d3103454cd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 4624, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1156, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 289, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_573a7ce37020be0d5682b46a1cdeb678(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 4624, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1156, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 289, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8274a87365116b30a0aa465c34145c1b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4624, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1156, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[289, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_53a2876a12f08f3c337c7e149c7a6109(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8274a87365116b30a0aa465c34145c1b
        def get_inputs(self):
            return [
                paddle.uniform([4624, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1156, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([289, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9e47fd261aca5e7ea9461aee526e9914(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4624, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1156, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[289, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6651f9386efeb26039495bf17c6e3d1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e47fd261aca5e7ea9461aee526e9914
        def get_inputs(self):
            return [
                paddle.uniform([4624, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1156, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([289, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0b276d84f7fa900e8fdfc4f8b02059b3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = -1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6069, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 6069, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e405fac3908016e8c4a3e87705092b16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b276d84f7fa900e8fdfc4f8b02059b3
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53a2876a12f08f3c337c7e149c7a6109(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8274a87365116b30a0aa465c34145c1b
        def get_inputs(self):
            return [
                paddle.uniform([4624, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1156, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([289, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28ec69a766c63dae62d5a6c6ca1625e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aeed443a0cef14b90625b6a17c4c06cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15910319983959198, 0.37697386741638184, 0.07013243436813354, 0.14398092031478882], [0.3346053659915924, 0.4726791977882385, 0.03578709438443184, 0.40225374698638916]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_43a962a8da1b4adc61d3a660e1e2d8dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41baac1cdf4b612b078c94271e932faf
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.015704719349741936, 0.1316700279712677, 0.0756133571267128, 0.4947950839996338]], dtype='float32').reshape([1, 4]),
            ]


    class TestPrimitiveOp_19545642fb3a8e976dc673d2b6d243ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3335762619972229, 0.255480021238327, 0.28426453471183777, 0.41945880651474], [0.26453742384910583, 0.30485397577285767, 0.11422444134950638, 0.10182178020477295]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_d9dc2631199e82390ce9203d6ae9ff5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8246e2d907d07186d6f3107f34b8108d
        def get_inputs(self):
            return [
                paddle.uniform([84864, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([21216, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5304, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1326, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([351, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89b46d3d6d7c8c314b59b41f5aeeb65c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9605db5d34190328a91cbc7b542e2f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 84864, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21216, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5304, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1326, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 351, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecfb6162af4062e6b19ff3d61e2b0835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc71b2b4ce912d6fc39c96fbb132300b
        def get_inputs(self):
            return [
                paddle.uniform([1, 84864, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21216, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5304, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1326, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 351, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa016ee8c58d3f3edb036cc6a434e27f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_e77884db9144be7b6bdc80dbe5b40353(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_193e276d4dc53c991a59f28d40dfa631
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_15e1bb4eb07f0acb575ac7271d6e95cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, None, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, None, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, None, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, None, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a26751b8da16f193be1939f38e853dce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15e1bb4eb07f0acb575ac7271d6e95cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd4052dda9813f2a3a4de7b22f2c6958(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9605db5d34190328a91cbc7b542e2f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49697fb732b95a83d76d3c21ae0a5cf9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17165537f396430d39f8755f4216b690
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14b18df948159dbd21151f11fe784cc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c832398a75ae36c54104ccd5d294e39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f79cb3eca39f5ecb7cdb672f686601d4
        def get_inputs(self):
            return [
                paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_118706aa20902e795ad43e1c3a8f48b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95654e1ecfb312059c706c08293f1b23
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd280f8ffed6350cde1d4616e34b7ed2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27a9e54228b95eaf48d93ede63917b56
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca698880727c8035f247522060ab4a85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d203906049cdc63985b4d54776d3a2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7c383e5b025ae30d8e609c0bc1b1bf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76192ba53b8ae9a3003a875d1e816a69
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6dac4538f3e283436077c377d02836ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef832da62bf203d07917fb50a433ab84
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            ]


    class TestPrimitiveOp_e8bfbf2192ebac1ef836d7168c9d2bcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd19aec86b16df7ce095b66011c39ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_db0f176757108902d8a18df945d48f67(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 15, 25], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_db06900798ed730bd313fd14a6bde0a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db0f176757108902d8a18df945d48f67
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cff2c8729695b18be3e1e946dc238d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_193e276d4dc53c991a59f28d40dfa631
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37eecf83334d0b05052f1b3d8119c6a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ffe0541ce9fc55854ab82f25de0b6932(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95654e1ecfb312059c706c08293f1b23
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_985ced31834c8cde77a9f99dc1a09625(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a946e19485dbdb17f72179b6753beb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c178d4c3fd160393d9c1f823beca52fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12f6631af91f636265712dbf46fc81e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29b9fa81183a10c24fc18b52c3142521
        def get_inputs(self):
            return [
                paddle.to_tensor([[4], [7]], dtype='int64').reshape([2, 1]),
            ]


    class TestPrimitiveOp_178f7495d58364f50dbcad58a954c2b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_688332b39c6b124768473e4ec975e07d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 2], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_ed5fcadcb2bcb4a336b2565af2ccd1ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05fa05daa6c838ed81a83ab37a187aed
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_dba3b02e6e14fcf6968e1277b994f5a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be9eb669746fd9491332d4aaee25f5ee
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06586539000272751, 0.08793290704488754], dtype='float32').reshape([2]),
            ]


    class TestPrimitiveOp_b5a1ee389ee7e4fde6d9c6b44e48a82f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddc81b56d4c2bac272e1ea55f518d5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef832da62bf203d07917fb50a433ab84
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            ]


    class TestPrimitiveOp_4c54f4d654dd1750fd613b06a5a32be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3ee04f2b25f862854073be798234eed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b207547ccceedf5026010f04b6e27536(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6dcac3f199c6692b1ec3b33123ecd39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90921eaea4d82de2c4f9271ad8876a65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b56ae0a4441e18c0da148f05f64f85c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76192ba53b8ae9a3003a875d1e816a69
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34fb740256cd5266386d5a040913c667(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2355c4a705b4d5e905360ff83161a2ab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 128, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 20, 128, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_265d3bc6cadc8843f9f652ba7a00a95f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2355c4a705b4d5e905360ff83161a2ab
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4374b27c3e6dd3ac2a21ce3bdb0cb17b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 40, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 40, 64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8e18aa709a55e09a8ebf0793e184fa7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4374b27c3e6dd3ac2a21ce3bdb0cb17b
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9ef5d74e604025b861b5ee47ce5da982(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 32, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 80, 32, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_09cc315b136ab79a164829899652e788(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ef5d74e604025b861b5ee47ce5da982
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b162f213c6335bafc864caa37ae48b36(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 16, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 160, 16, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_01ef345502931f4aa76449de66d554d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b162f213c6335bafc864caa37ae48b36
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48063358ecc5c50a0fb68408d5ba3d2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c178d4c3fd160393d9c1f823beca52fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 92, 92], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39946cc956b708d876c076d161dd5c98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c178d4c3fd160393d9c1f823beca52fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71ee847790586da695474af0dad9ef74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b11311905511fba2d3192dc762a732dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1304a66b60b833af089f5f14d3a876a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8246e2d907d07186d6f3107f34b8108d
        def get_inputs(self):
            return [
                paddle.uniform([139392, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([34848, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([8712, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2178, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([561, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_316ef40e8385bf7d8ff59b966c1c89a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9605db5d34190328a91cbc7b542e2f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 139392, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 34848, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8712, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2178, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 561, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8035fc68f98621e8afa9193738ffc159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc71b2b4ce912d6fc39c96fbb132300b
        def get_inputs(self):
            return [
                paddle.uniform([1, 139392, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 34848, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8712, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2178, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 561, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f89e11afac043fb4e3fd045ba4ce53be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2215ccf5f23acfd3d28c83780f21e996(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b777cea8a3d903a6af3695f077a709a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9cc50f5c1192de53492493051599d446(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_2cca0a9abebd7441bc16fcbb8115c174(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b777cea8a3d903a6af3695f077a709a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_598ea79856b91512c97beb0219127aa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27a9e54228b95eaf48d93ede63917b56
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af553e21d28e7b705134006575d7eea1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9935ee3d91241c6a20b1576b03369784(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 180, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 180, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_69fa7f3a5c523e10afdba33fee006b65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9935ee3d91241c6a20b1576b03369784
        def get_inputs(self):
            return [
                paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a6a4b9d4215805e519afadea5fee41d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df2ad9edf873f0a891184aae3978c449
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1703c71f875927834cc13b431229fc00(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 2
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 4, 50, 76], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4, 1, 50, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_51d6a8b8278c902121a52874daf57337(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1703c71f875927834cc13b431229fc00
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4, 1, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07bcb584c04c06392c59aa4e1af55d43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 46, 46], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ed6def8a528dfb962ce3615e8adee813(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[15200, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[3800, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[950, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[247, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[70, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a2186bd7a693f37578718a98e7d8a8d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed6def8a528dfb962ce3615e8adee813
        def get_inputs(self):
            return [
                paddle.uniform([15200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3a6befd455865c679edd079105683bfa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[15200, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[3800, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[950, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[247, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[70, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d50c99081a40fc5d59e7d66dd6c96a34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a6befd455865c679edd079105683bfa
        def get_inputs(self):
            return [
                paddle.uniform([15200, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_720465c36c133ac574ac0cefa0991333(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[15200, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3800, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[950, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[247, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[70, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2dbd6e9754caad7b2ec1d614bb8cb8ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_720465c36c133ac574ac0cefa0991333
        def get_inputs(self):
            return [
                paddle.uniform([15200, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f437b74811e03a453ef693a5ca4ed3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdc2c6deaf146636dedbc47fab21587f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d203906049cdc63985b4d54776d3a2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86d5b00d1f1fa03c945a6a1f39d2d697(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8246e2d907d07186d6f3107f34b8108d
        def get_inputs(self):
            return [
                paddle.uniform([163200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([40800, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([10200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2550, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([663, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ce3684cc110f9ce2901934d25aed39a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9605db5d34190328a91cbc7b542e2f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 163200, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40800, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 10200, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2550, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 663, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_063b42d7d5921ca5f9e7b2202ed0c1b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc71b2b4ce912d6fc39c96fbb132300b
        def get_inputs(self):
            return [
                paddle.uniform([1, 163200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40800, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 10200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2550, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 663, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8d0c29620ac1c2ccb81ba4c76afc903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17165537f396430d39f8755f4216b690
        def get_inputs(self):
            return [
                paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da70edbc3185f6d948d57d704960c9b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17165537f396430d39f8755f4216b690
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 84, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96f604ab75d2693ab35a29c61860325a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a53c6e88ae1886f78055aee4643423b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4163cbee4349c5805c670b391579e50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38080445be453a6914e8f74ade8c5daa
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9e8d71d651c2cb96b1fc1a6f42cda05a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7a455644d280efa83516ff6e202e978b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e8d71d651c2cb96b1fc1a6f42cda05a
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc19d05fb0577412f98d945ebabbb82d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ffbe35993479fde103f94ac2c55b2e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f36a79031d19f9ade22484618626010f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2f4ab04a3e148501b4f83f9b6318128
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_573850a60a6ef2f9d2adae25a94cf45b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b6609bdd6a9299b0fd68f2a1b9494f4
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1046d043ab1317dd0db2f9321c2b24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_193e276d4dc53c991a59f28d40dfa631
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d11526cb5bc350f8ae679a8ba55aeb18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a7769b4a30275dd5f81e65dd2ddc612
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_755bbacf733992cee38774654b77fcbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4aded8307ddca406fe4821e8c0273fd6
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_52f05524b007a3215882d484fef9dc2d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1bcd0539181003ee015346746c44b4fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52f05524b007a3215882d484fef9dc2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6dab6c64639f004c00937270049bc7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8246e2d907d07186d6f3107f34b8108d
        def get_inputs(self):
            return [
                paddle.uniform([92928, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([23232, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5808, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1452, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([363, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12b7f1c9edc732221beee3aa40513820(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9605db5d34190328a91cbc7b542e2f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 92928, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 23232, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5808, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1452, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 363, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8876955d0d6c9b925750e2c2508ddb6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc71b2b4ce912d6fc39c96fbb132300b
        def get_inputs(self):
            return [
                paddle.uniform([1, 92928, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 23232, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5808, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1452, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 363, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd4c5bd137e8f01533eb22e121b81e43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461d923546c9e1b9f6d41dea496e3e06
        def get_inputs(self):
            return [
                paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40bda44d0d5339b4fb251f089cb7b6d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_785362186c98070dc9d40614b070c052
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9da149f97dae43e63c042dda67e3b4a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_718973585250f3f3df97d2287c80a022(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0cbbb0d3529c8af89479d6c1c894b658
        def get_inputs(self):
            return [
                paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1cf341349f9f201667f5f538a5079a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e9a6d59e6e3204394e72adda01ebb04
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ddef82572a6c0ca6e8ab4b022ef15487(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 150, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ffd8068d9929f18c4145fdb7ec5f00a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ddef82572a6c0ca6e8ab4b022ef15487
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 150, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6dac4538f3e283436077c377d02836ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef832da62bf203d07917fb50a433ab84
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            ]


    class TestPrimitiveOp_75987329082a2e95e9926f1993d28212(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59c83a5ec9bb2b16eacd992488268797
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6912ee2adeec91822688b63c5214800d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 116, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 116, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f135f3be14a638d02300445feaa8e46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b668d235f7eedf6d53a33e3a424654b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81842be2a57e4a1ba5fbd915505dd029(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fa590ed4b72d434fdbbcc1fc8f86e73
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_790365952124290c9715da048c058911(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c178d4c3fd160393d9c1f823beca52fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73df2b45b3170384087d2ef105ce829e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b6609bdd6a9299b0fd68f2a1b9494f4
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5373e9e13a9525dbc4a44cf84606954c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2c24cd2b46e3808ce8a84a5dbecea10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461d923546c9e1b9f6d41dea496e3e06
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 120, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 36, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b1e06f75e2e92a51b579b485ed6a6ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_acaf03e16abdb01a1086da742172f60b
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_815fd11ef3cbe2802c4a7209111825d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8246e2d907d07186d6f3107f34b8108d
        def get_inputs(self):
            return [
                paddle.uniform([154560, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([38640, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([9660, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2415, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([648, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_665bb7ff1a92596a97ffdfda8612817a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9605db5d34190328a91cbc7b542e2f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 154560, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 38640, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9660, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2415, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 648, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_286db878b2f5ba14f8f7d371e00682ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc71b2b4ce912d6fc39c96fbb132300b
        def get_inputs(self):
            return [
                paddle.uniform([1, 154560, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 38640, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9660, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2415, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 648, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed92518122fc7c90315ac9dd5e714f68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4aded8307ddca406fe4821e8c0273fd6
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed92518122fc7c90315ac9dd5e714f68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4aded8307ddca406fe4821e8c0273fd6
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeb191664b11ca61b03bf5f87354cf59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf6a436ca86e7216168f4d3af16cd652
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaaba8e617b2a440328152519cb4645a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf6a436ca86e7216168f4d3af16cd652
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_597957f514a959b073824e774d019b19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_597957f514a959b073824e774d019b19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d022c6b0b66f7ddb0d52ed5a97ca86
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46d0e4dd6e81cdaa16ed0ece58d8a8e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42b1750dc4443c8c7dadf5813b1237c7
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_52d2832120c6ac690390545a4dde1b48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42b1750dc4443c8c7dadf5813b1237c7
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39d51fd53728f510a52fcb92f5dc887c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c178d4c3fd160393d9c1f823beca52fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e129c953d9fbe51717221bb8768f9361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e87959ec12b4b84b2b8766c0d176f33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 224, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 128, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 128, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 128, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 128, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5eb45f6e9559f83f5a7315a90c0f8804(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdea871a60d6fba5929b463d6ea8ed83
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_802de5501aa339657dc493ecb92e5361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea87c44da5f8b90619ca4475f041774d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_2c3099449f7389b1b5445018eaf10834(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0cbbb0d3529c8af89479d6c1c894b658
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 256, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 60, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a479a146299e0bf8703aa7a5bb61ffb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76192ba53b8ae9a3003a875d1e816a69
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7206f60de8e9367167c0e989cd835e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b11311905511fba2d3192dc762a732dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f186e63e4cd4790d8febccdf2298de92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27a9e54228b95eaf48d93ede63917b56
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3a304b19325508acd5a9fbb74235728b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48, arg_0_49, arg_0_50, arg_0_51, arg_0_52, arg_0_53, arg_0_54, arg_0_55, arg_0_56, arg_0_57, arg_0_58, arg_0_59, arg_0_60, arg_0_61, arg_0_62, arg_0_63, arg_0_64, arg_0_65, arg_0_66, arg_0_67, arg_0_68, arg_0_69, arg_0_70, arg_0_71, arg_0_72, arg_0_73, arg_0_74, arg_0_75, arg_0_76, arg_0_77, arg_0_78, arg_0_79, arg_0_80, arg_0_81, arg_0_82, arg_0_83, arg_0_84, arg_0_85, arg_0_86, arg_0_87, arg_0_88, arg_0_89, arg_0_90, arg_0_91, arg_0_92, arg_0_93, arg_0_94, arg_0_95, arg_0_96, arg_0_97, arg_0_98, arg_0_99, arg_0_100, arg_0_101, arg_0_102, arg_0_103, arg_0_104, arg_0_105, arg_0_106, arg_0_107, arg_0_108, arg_0_109, arg_0_110, arg_0_111, arg_0_112, arg_0_113, arg_0_114, arg_0_115, arg_0_116, arg_0_117, arg_0_118, arg_0_119, arg_0_120, arg_0_121, arg_0_122, arg_0_123, arg_0_124, arg_0_125, arg_0_126, arg_0_127, arg_0_128, arg_0_129, arg_0_130, arg_0_131, arg_0_132, arg_0_133, arg_0_134, arg_0_135, arg_0_136, arg_0_137, arg_0_138, arg_0_139, arg_0_140, arg_0_141, arg_0_142, arg_0_143, arg_0_144, arg_0_145, arg_0_146, arg_0_147, arg_0_148, arg_0_149, arg_0_150, arg_0_151, arg_0_152, arg_0_153, arg_0_154, arg_0_155, arg_0_156, arg_0_157, arg_0_158, arg_0_159, arg_0_160, arg_0_161, arg_0_162, arg_0_163, arg_0_164, arg_0_165, arg_0_166, arg_0_167, arg_0_168, arg_0_169, arg_0_170, arg_0_171, arg_0_172, arg_0_173, arg_0_174, arg_0_175, arg_0_176, arg_0_177, arg_0_178, arg_0_179, arg_0_180, arg_0_181, arg_0_182, arg_0_183, arg_0_184, arg_0_185, arg_0_186, arg_0_187, arg_0_188, arg_0_189, arg_0_190, arg_0_191, arg_0_192, arg_0_193, arg_0_194, arg_0_195):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48, arg_0_49, arg_0_50, arg_0_51, arg_0_52, arg_0_53, arg_0_54, arg_0_55, arg_0_56, arg_0_57, arg_0_58, arg_0_59, arg_0_60, arg_0_61, arg_0_62, arg_0_63, arg_0_64, arg_0_65, arg_0_66, arg_0_67, arg_0_68, arg_0_69, arg_0_70, arg_0_71, arg_0_72, arg_0_73, arg_0_74, arg_0_75, arg_0_76, arg_0_77, arg_0_78, arg_0_79, arg_0_80, arg_0_81, arg_0_82, arg_0_83, arg_0_84, arg_0_85, arg_0_86, arg_0_87, arg_0_88, arg_0_89, arg_0_90, arg_0_91, arg_0_92, arg_0_93, arg_0_94, arg_0_95, arg_0_96, arg_0_97, arg_0_98, arg_0_99, arg_0_100, arg_0_101, arg_0_102, arg_0_103, arg_0_104, arg_0_105, arg_0_106, arg_0_107, arg_0_108, arg_0_109, arg_0_110, arg_0_111, arg_0_112, arg_0_113, arg_0_114, arg_0_115, arg_0_116, arg_0_117, arg_0_118, arg_0_119, arg_0_120, arg_0_121, arg_0_122, arg_0_123, arg_0_124, arg_0_125, arg_0_126, arg_0_127, arg_0_128, arg_0_129, arg_0_130, arg_0_131, arg_0_132, arg_0_133, arg_0_134, arg_0_135, arg_0_136, arg_0_137, arg_0_138, arg_0_139, arg_0_140, arg_0_141, arg_0_142, arg_0_143, arg_0_144, arg_0_145, arg_0_146, arg_0_147, arg_0_148, arg_0_149, arg_0_150, arg_0_151, arg_0_152, arg_0_153, arg_0_154, arg_0_155, arg_0_156, arg_0_157, arg_0_158, arg_0_159, arg_0_160, arg_0_161, arg_0_162, arg_0_163, arg_0_164, arg_0_165, arg_0_166, arg_0_167, arg_0_168, arg_0_169, arg_0_170, arg_0_171, arg_0_172, arg_0_173, arg_0_174, arg_0_175, arg_0_176, arg_0_177, arg_0_178, arg_0_179, arg_0_180, arg_0_181, arg_0_182, arg_0_183, arg_0_184, arg_0_185, arg_0_186, arg_0_187, arg_0_188, arg_0_189, arg_0_190, arg_0_191, arg_0_192, arg_0_193, arg_0_194, arg_0_195]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_831c9c5d8aa9488730c4ff9422a32347(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a304b19325508acd5a9fbb74235728b
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef5c6e355cb6553298f12fbd7e8ae7ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d408e5a64902e2c528e3bc4db950c9c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 200, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c0cb295a169a74a462c95cefed64f17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b777cea8a3d903a6af3695f077a709a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dee36ac5f6ab9de77b635a7937b221c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8246e2d907d07186d6f3107f34b8108d
        def get_inputs(self):
            return [
                paddle.uniform([165888, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([41472, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([10368, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2592, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([648, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e0a67fa6c6606973fde1f4f29f0ec2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9605db5d34190328a91cbc7b542e2f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 165888, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 41472, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 10368, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2592, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 648, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8febe6d36a214b22a2c231a41c0919be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc71b2b4ce912d6fc39c96fbb132300b
        def get_inputs(self):
            return [
                paddle.uniform([1, 165888, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 41472, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 10368, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2592, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 648, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70468b99a2e60a5c01db2c12f9b28967(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 5184, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1296, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 324, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aef8e1e9d8bf6c57d0393c12c79e392f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 5184, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1296, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 324, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03ab6a91ec6481d5bf96159639c5c485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 5184, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1296, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 324, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_365b77ef2e17db7e275fa899a538e362(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5184, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1296, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[324, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd6580c71826af54bd38d6109d7a0fe8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_365b77ef2e17db7e275fa899a538e362
        def get_inputs(self):
            return [
                paddle.uniform([5184, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1296, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([324, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d8a8ffe35700f3a7c6f1b644f3c94087(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5184, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1296, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[324, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_59ad5b41836c42204250ea8e85dcbf25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a8ffe35700f3a7c6f1b644f3c94087
        def get_inputs(self):
            return [
                paddle.uniform([5184, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1296, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([324, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eab3b476f49d8dc2d871e8520a7e64f3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = -1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6804, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 6804, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dde87ab8df8d1e712c4083ab63b903d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eab3b476f49d8dc2d871e8520a7e64f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd6580c71826af54bd38d6109d7a0fe8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_365b77ef2e17db7e275fa899a538e362
        def get_inputs(self):
            return [
                paddle.uniform([5184, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1296, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([324, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cff96cfe0edb067592a0819abe086f7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b777cea8a3d903a6af3695f077a709a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc8987d0fa81c1bfdd44ed683430ae39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
            ]


    class TestPrimitiveOp_a357c84dfbd5403362aec51a533f5e98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cedba2eb201a3db429333bd7ed9adbfc
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e13b3cace1eab17769e3d996ba8d5aba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4b41ce4ef6547665fd203035361f91fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_96225f5e01fe0b8f1407ce386367a1ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b41ce4ef6547665fd203035361f91fb
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8ead674f0e8a8de8f1ac844c2c891de9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_47d1d41757fcf3250677c0a351f010de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ead674f0e8a8de8f1ac844c2c891de9
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59cfde7475aca114a1c43c03c6d40b6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_193e276d4dc53c991a59f28d40dfa631
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5380655c11b9670c326f6862b8eeac5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b6609bdd6a9299b0fd68f2a1b9494f4
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_76f60610aeda5413048305e50880dcd7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 2
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 4, 25, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4, 1, 25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1beaaead76af464192e0eb00633d2b92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76f60610aeda5413048305e50880dcd7
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4, 1, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a949594e1ed2f9f5b40cd8242250bb28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4aded8307ddca406fe4821e8c0273fd6
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34b4a076f6ae54d7fa25d90811660483(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b6609bdd6a9299b0fd68f2a1b9494f4
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6251ab9e87fdc4d30972581fa39b3b45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b777cea8a3d903a6af3695f077a709a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5f7e3fb3bf0f6a1195e723fee0ebc66b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 2
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 4, 7, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4, 1, 7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8264163c018726bebffc2679974087a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f7e3fb3bf0f6a1195e723fee0ebc66b
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4, 1, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_deda542b498eb303c3a11ebc6dbcd8cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 58, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 58, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddae93e2b2eb403527523761718d236c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df2ad9edf873f0a891184aae3978c449
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3b0cecf503832cde54979db6d853720(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e35e1799178626f7263cfaba0c944d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68ff6fa7edcd06bc9242df895016d196(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc19d05fb0577412f98d945ebabbb82d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ffbe35993479fde103f94ac2c55b2e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f36a79031d19f9ade22484618626010f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2f4ab04a3e148501b4f83f9b6318128
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0cc2f0d23c249eee4a7dcb537d51fb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2633fe2a6c96ee8bbaee43b6280eb112(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 2
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 4, 100, 152], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4, 1, 100, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c8d398b8749c770ccded4aa686f9a1cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2633fe2a6c96ee8bbaee43b6280eb112
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4, 1, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41a443ae64993eaa191dbbfc44f2929a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 512, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67492b5f94de583300cb8f548db3c33b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9013641300a6f9d6434cf98fb91a01a7
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_80ef0a3207b83acf327a48e959af3d08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461d923546c9e1b9f6d41dea496e3e06
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 256, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 36, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b3742fa4bab82599fa10911274c8241(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef832da62bf203d07917fb50a433ab84
        def get_inputs(self):
            return [
                paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            ]


    class TestPrimitiveOp_1b5f014cee568785cef2ef7e944e2690(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_03d3f66dde9718a8e09c67fea644770a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = -1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0d7959346cc78c605729554144d6e1d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03d3f66dde9718a8e09c67fea644770a
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81197f36d187bfcafcc94bc9ab699c3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_112bb61e9955d212d60f0d3b3dcb13d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_52ffbb8ae4c8236f3a3188b26050dd41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 448, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5dc69633ef838b624f2dc23bc783f6f6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fbcd95cfa099b5a1a57d2915fd27ce11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dc69633ef838b624f2dc23bc783f6f6
        def get_inputs(self):
            return [
                paddle.uniform([129024, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([32256, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([8064, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2016, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([528, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eb7b3b8365c3c6768dfd2f82da15879a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 129024, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32256, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8064, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2016, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 528, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f6991116c10c5e51bcd208851955aea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 129024, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32256, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8064, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2016, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 528, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_899c6f506ffb34c52f62e183d8304069(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_926e07052c6493355b6361030104c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_cc0ccca8879b60cf9f5fe9b1b7a5a369(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 152, 152], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 152, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c52c037c2bfc05b1401bbc6546a24eff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c408233a0001a4152195e80282234d46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5361378694a31c6ad75ff8a4ada1f17d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4568cc28a5070b7a144c20d81a279f30
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4726118743419647, -0.5898312330245972, -0.09625720977783203, -0.42435988783836365, -0.17186817526817322, -0.5406213998794556], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.3495945632457733, -0.4441525936126709, -0.45715004205703735, -0.41611766815185547, -0.6559632420539856, -0.6102994680404663], dtype='float32').reshape([6]),
                paddle.to_tensor([0.5749874114990234, 0.8079026937484741, 1.0657973289489746, 0.878547191619873, 1.0153841972351074, 1.0460015535354614], dtype='float32').reshape([6]),
                paddle.to_tensor([1.1797178983688354, 0.674731969833374, 0.7607995867729187, 1.0119351148605347, 0.775779664516449, 0.7970899343490601], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_f5dde48b05a820fb588b7839ab485e9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_69d91cd8e75d0f4d797a5e46b01b2351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_a0c498776046c4a4351f5cfeaad4cffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ad9dffad327cd3a8be8dda06858f460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dc69633ef838b624f2dc23bc783f6f6
        def get_inputs(self):
            return [
                paddle.uniform([115200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([28800, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([7200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1800, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([450, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_396f3d0f4bb5b0a6b86266960e5de59e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 115200, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 28800, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7200, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1800, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 450, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c5a9d6f49b920c7d1ffbf89d33de333(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 115200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 28800, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1800, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 450, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4734ce31d43c9cb46ee820a48f86fa30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0929a44e57036e7e527ff59bfee1599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be08ff495a90b6c0d8b8d8aa53c983bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09dbe901a54cd57fa2c471870b04179f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_ff0e1ccf75c74f3dffab3011974a9e61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 104, 104], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 36, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53bdca45d2c7b094d86feadd6f0b223a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 60, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb0d258ca0e89ef345e0f2dbdd6e13b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d462a02270c2baa508cfcc78564bb95f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1fe22ba1086336900429993e3ce79f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8eb1c3bc24e21f64d2b119df16f5c3dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eadc7eb69b5f4e5ee3105b43519477fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8eb1c3bc24e21f64d2b119df16f5c3dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 150, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ca00c57df3314141f24a750357608d55(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0):
            input_0 = [arg_0_0]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8b78014c0f62c0a690bd2ef5734a5393(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca00c57df3314141f24a750357608d55
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
            ]


    
    class PrimitiveOp_1328401f728849a715eaaaf50b1b739f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 2
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f6a6531be8d5a875f3b37f228fdb22bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1328401f728849a715eaaaf50b1b739f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_5032b62db8b5a266ece5b2d7abfc67ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 9216, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2304, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b6c94b883c336c03204c6489105f6a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 9216, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2304, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5b3d1876bef63976853e8297ade6981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 9216, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2304, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d3690a95e8d928803eb155a64110edfc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f5313fdbb16e18196a208a7ddf065d00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([9216, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([2304, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([576, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77146d17b7d2ce8959066d7a01e0640b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([9216, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2304, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([576, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_347d3c328b432bc92216680b2eebae3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5313fdbb16e18196a208a7ddf065d00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([9216, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([2304, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([576, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_850e0685ada8e02d4538668543c91319(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d97f132da26f03dba99596e6c058d0df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81dd5f001d60335c874b1dbb2475a183(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e82f8b0dcbe1389173792105a097c3e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69d91cd8e75d0f4d797a5e46b01b2351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_1eec25ec0cc270afa62f00ed11b1b15d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08b753b83daffe5eba763b0912ede168(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0480e2ef5ee955cb43284174da5dbc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc3d6ba772aba40377797242c2c054d4
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 48, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 48, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2144a2b2e156b8135874f5268ca80387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
            ]


    class TestPrimitiveOp_b424678579ec88e44c12ac7fcca78b57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9158a44d17262b9cd163ff04f7b3acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87871265420f016b916dc2496a32a483(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b78014c0f62c0a690bd2ef5734a5393(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca00c57df3314141f24a750357608d55
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_f6a6531be8d5a875f3b37f228fdb22bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1328401f728849a715eaaaf50b1b739f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_7bed22811e267741ec78bb568bea3b22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab7d35bce4cad2104e770b8614d04c50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
            ]


    class TestPrimitiveOp_5a436f7782805d976a224d8f25ae39ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_476ea7c27b97cd23d0291a352d1e12de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4090d7a989653bbb6c18a72c5cea596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cade650ebb48afe3bb584f2e5e5864e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 34, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fcf0ad60b6ad8e4d0bba82403f8b6d31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 36, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 160, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69d91cd8e75d0f4d797a5e46b01b2351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_798804ddfda289f5a3ee96c3865f6c2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cda5b9e05a58f268b3218ea23cc069f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
            ]


    class TestPrimitiveOp_65b894ba393d1585155794b72531cabd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba8f85b7b4609aef3a7ba7a40af226fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5dde48b05a820fb588b7839ab485e9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_d6dcac3f199c6692b1ec3b33123ecd39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1015e47a7883f59812b13b36c4917d22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbeb25ed3ac15e2a7558ab1c6c2694a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c75d6f25a1fe3f9e0a191e59ffc75b91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f882eeddb520ec85e1d27493f6c4db30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_985ced31834c8cde77a9f99dc1a09625(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61396c6cdcd467e3895fdb800ef8a114(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
            ]


    class TestPrimitiveOp_57a4392415c235c02b2c53143177f7e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9b65f9754490c5650e88889e80ecfb56(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_adbd0d70685e5fe220464b07873b1761(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b65f9754490c5650e88889e80ecfb56
        def get_inputs(self):
            return [
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9521ced8cb5169531afaee6f17b85f4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_981733171efe7cdfeb59f4f4e9b3a07e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29b9fa81183a10c24fc18b52c3142521
        def get_inputs(self):
            return [
                paddle.to_tensor([[4], [7], [8], [2]], dtype='int64').reshape([4, 1]),
            ]


    class TestPrimitiveOp_e4bf4329285749c98b247b3989d7b2eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_00a6fffbd2de9c190c75be6d551b8399(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([9, 9, 3, 6], dtype='int32').reshape([4]),
            ]


    class TestPrimitiveOp_ad9034741bd1b9f102d001ad5a57b869(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05fa05daa6c838ed81a83ab37a187aed
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_f95b1c4b242e44d30abfd224c19819b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be9eb669746fd9491332d4aaee25f5ee
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2847067415714264, 0.4610000550746918, 0.04074212908744812, 0.044464848935604095], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_0189e1a24f624287703324bd36c0d53b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87984a79832b96df9635e15a55b517db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 60, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_426514ca536d2838ba41f576fbc761de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29b9fa81183a10c24fc18b52c3142521
        def get_inputs(self):
            return [
                paddle.to_tensor([[4], [7], [8], [2], [2], [9]], dtype='int64').reshape([6, 1]),
            ]


    class TestPrimitiveOp_632a85866daac99cf6111d95750a9188(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([9], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e5fd1908a1cc3d4eb4f76e3a4b964b79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([3, 6, 3, 3, 1, 7], dtype='int32').reshape([6]),
            ]


    class TestPrimitiveOp_d8e36e5f8dd3adf06fa3925dfe008f7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05fa05daa6c838ed81a83ab37a187aed
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[6, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_cb09cb162a45d374ecca4f6d5df1397d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be9eb669746fd9491332d4aaee25f5ee
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3677123486995697, 0.03161490708589554, 0.4054182469844818, 0.1435633897781372, 0.4872145354747772, 0.4321914613246918], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_b405b8fba99e6482b1afce71d9b01db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 19, 256, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd84921fed254b3f4e02910b33ee5cfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_667c2516da755eebfa55d0125d7dbe78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
            ]


    class TestPrimitiveOp_a0c498776046c4a4351f5cfeaad4cffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e35e1799178626f7263cfaba0c944d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b64720a0b46675b0fd5bb8b981a47f13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e48fbe58db19d9ce3164b462f1fda8f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f54fceef130d50e8e6789b59cf136331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe7f258cab02450e3ffb4225afa91410(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([242991, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23a17d698c0fc03ad96cfeb1de6001a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce434da45bb5b39d0df9b5f4f1c57fb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1024, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_52fc7ea0f365f29676573ef33a88d3a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1024, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cb7254f5ace20d05471ae6649565f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2653a2d9011f714c734f2eee5835514f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([4096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d7efb41e7a56bf998a8545541489f99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([4096, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69b2ead09dbee8be4ade7f0d752a8d70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2653a2d9011f714c734f2eee5835514f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([4096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cff3bb13445432acde7ce3c98c749ac1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_2ba79a401113e7da1277be0e2b206321(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_849ec0c0aebc15b4e68b00d7bfade008(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a173a0368969c28e1c921696193878f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2712c3a971e865d9a08d3918c49f7600(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f948059304cf7ee07d286af6e68650c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0c498776046c4a4351f5cfeaad4cffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bb0761ff7d764542255b9f8264dd04e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 96, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_160ae8bdb6262ad4366aa202ee661263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8eb1c3bc24e21f64d2b119df16f5c3dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6eded2b9f0650d97fa6423c7db131455(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 224, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 128, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 128, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 128, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 128, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4beb79ea56326cbd79d002f14167e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
            ]


    class TestPrimitiveOp_a0565095165bef0c6fc137fd2d0eaf5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4568cc28a5070b7a144c20d81a279f30
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4631315767765045, -0.3884684443473816, -0.7395162582397461, -0.6278271675109863, -0.1999669075012207, -0.19013196229934692], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.5041857957839966, -0.5462598204612732, -0.20084482431411743, -0.3272498846054077, -0.5095967054367065, -0.5581246018409729], dtype='float32').reshape([6]),
                paddle.to_tensor([0.8244056701660156, 0.9471744894981384, 0.8348401784896851, 0.64537513256073, 0.8654975891113281, 1.1398656368255615], dtype='float32').reshape([6]),
                paddle.to_tensor([1.0452922582626343, 0.6076709628105164, 0.9019972681999207, 1.311782717704773, 0.950197696685791, 0.6846595406532288], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8732e3e25b3222096bd2cabf65fad963(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6a71d8e7ce47d8989dfc6a4126b42e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc3d6ba772aba40377797242c2c054d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53284743c95f76a266b5387087301134(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4dc1b1cf2e21356fef7b35f006023514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_576d0f4b0cc4c0987bf4031d8b112e0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96f7074ba1d88e0ae8be6dc870a790e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_274a4a3f403c685000e3e4a100c41d86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            ]


    class TestPrimitiveOp_49b69cef6cc15224dfd0e5d2aac8df3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8365451f8a01ff0cdd9bcc0b782e520e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a173a0368969c28e1c921696193878f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([171888, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e684593b505c24296028c5f6b96bf04c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c735b2e1fd417b767cb42a8b6535d18a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 48, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a63d36e4c21637f6f4105b4637c8178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30aeb9279f619b5efb17e802a80ecb85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30aeb9279f619b5efb17e802a80ecb85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_483c0be374ecb90675c6c10d4b495801(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e7ff631e48c6cbb9d46834da6a4d841(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcbed5168bb9b9aa2c780c8288d3192d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcbed5168bb9b9aa2c780c8288d3192d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3df2baa0b0945f291d8e058e8a7ee32f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc847bdd1474d9dbecbd05d961bfce83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29af46fc00e9b09d8bcdab8fe1a95bab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([3, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_0156216e1d31086e895642528eb0140d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e79d4d9124303baca93465c8e478adf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 160, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_925e82b20a30f438960a717142d3715a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b3781a707e2b431f11cb70d67b1bdbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dc69633ef838b624f2dc23bc783f6f6
        def get_inputs(self):
            return [
                paddle.uniform([139392, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([34848, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([8712, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2178, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([528, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6149bd402fa18fc4d37f900266b45b45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 139392, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 34848, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8712, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2178, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 528, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d36f2bcd9922adc55e765efffd0edc36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 139392, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 34848, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8712, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2178, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 528, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a496bc4f416591274b8dbf35086ac9e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9168911e0bad921618c38c3b01466598(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec945ea9b6eaba813fdfd11b0e31c0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8638b9ffdfb34bb136bf8d68d607cd79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 512, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 160, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39789085aad91d8f8b6fe8aa61923ee3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 168, 168], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 60, 168, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3cf1c94b3a0881eac38a308b04db8bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([217413, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_374727ee74c121edc1c7e9134332071f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 6], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_7cbe129867c1c5f3e716bad8bc1312c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4177cbcec63c69fabd02bc7d02a0199f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a11e5df6ef3d2e8028afb33695e002f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a11e5df6ef3d2e8028afb33695e002f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8688cfa3766b77c226ab7b04e3dded5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dee48b40f99febbbdd0136e9fc179a1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9853a7f0febb83db26bd325c0d07dc2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9853a7f0febb83db26bd325c0d07dc2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59152ad1094900e0b27402ef9a59711f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_adfd16bab70cb487ff794bf0d51c0c8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b1c7d51f52e6f651974ddbd4c599182(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d2b023da1a9456d668deee2d9825f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cfc93308792ded2d4df422b37326b103(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 160, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 34, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ab0b77a5729694f8e642b272525f6dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4f9a0f0f628efd8d407d496e3ce1acd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0c498776046c4a4351f5cfeaad4cffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64259c1a3bc05e5e69c8d543e00a33db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 448, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a73d0ccd91bfacb12c307f63ed5d228e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 28, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 36, 28, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69d91cd8e75d0f4d797a5e46b01b2351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_942e474a24ac25a999762145a2405c78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 76, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5373e9e13a9525dbc4a44cf84606954c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5f88e839eb6d281035b18ff4daded60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            ]


    class TestPrimitiveOp_b4b49a0fbb1abbb1205812789caea86b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0fe198348526e8d1727bfa721d82669(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([6, 9], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_d65e52eb886e24bd74d96352fa63c740(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c39b6475e5ca9adcfbff590dc5fd9ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 6400, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1600, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 400, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5736a577117faafb8551cb67a367ea67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 6400, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1600, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 400, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d8443db498b0d616a808784bdedcb2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 6400, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1600, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 400, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56d376f1cb215a0242dfa67cfcce40ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([6400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1600, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f641b39bf8b749e46bd3040aa32578ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([6400, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1600, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([400, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e35e1799178626f7263cfaba0c944d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56d376f1cb215a0242dfa67cfcce40ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([6400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1600, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([400, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f7d3be4d352571e6b0b689556243d71(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b9cb72001bf04e50ca80aa3fb8f48eaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f7d3be4d352571e6b0b689556243d71
        def get_inputs(self):
            return [
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1062f61453b86a768183a51a19cdf1ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc3d6ba772aba40377797242c2c054d4
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ea1b98075328286a1bc06c72a75a3c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_802caa5c66bca80180e24fcd38407a72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
            ]


    class TestPrimitiveOp_4734ce31d43c9cb46ee820a48f86fa30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0929a44e57036e7e527ff59bfee1599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad423b898367571a15a9ef62c86d4c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 36, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3ee04f2b25f862854073be798234eed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f579ffdd1f269b7810026e3cf3962e14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e92e170f44df8705b138040ebc8a899(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6587e492f41cd2b1da73db9158d63401(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb46f981a5b56ae02c98f51b30c7f029(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([7, 64, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 64, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 64, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 64, 7, 7]),
            ]


    class TestPrimitiveOp_f7f483617fcf11d2476f906358ab84c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 136, 136], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 136, 136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f40fb506b24eb037e4803a8c6afdb73c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5f88e839eb6d281035b18ff4daded60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            ]


    
    class PrimitiveOp_80a7c0d8632d388263ab6994bde0ef15(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 2
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1580a9cb5026e84d3de97be439d908a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80a7c0d8632d388263ab6994bde0ef15
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 13, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4, 1, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8077c76e6a511e6b3ac137ca91026a48(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e8774b8884c914582806a30fcd1cbbef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8077c76e6a511e6b3ac137ca91026a48
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_ba83a50087bf38238f62c6b142be9702(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c1f4033ef0eb655ff6ce1739e7eb418b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba83a50087bf38238f62c6b142be9702
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 2, 3, 4], dtype='int32').reshape([4]),
                paddle.to_tensor([0, 0], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_d023b2dac6266d44330d7a40a1a5c59b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5dbd369357b7d75c5c67ee4291ebb7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc3d6ba772aba40377797242c2c054d4
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e31d67df109f23c734a18310ce61a208(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d49e27633eb8b2b73ebb1b9d1aba0430(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dc69633ef838b624f2dc23bc783f6f6
        def get_inputs(self):
            return [
                paddle.uniform([182400, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([45600, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([11400, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2850, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([741, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3703340f1ed17543aa8e075418fcd61e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 182400, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 45600, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11400, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2850, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 741, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_669e02f5a5d4d30915409a1df7bcdcda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 182400, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 45600, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11400, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2850, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 741, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ea7e373d96a6bd2637a713058d896c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([86970, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_861c1ea1a563a4232b2fad5de2ea6a06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([205923, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ffdf1268bcef1ac380e7171bcd2d9642(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_637d4d4534aaedd7117317c9514f8f79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_515c0d132679c20ea18ffc505519b787(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
            ]


    class TestPrimitiveOp_bed0b408f2530799bbce2286bf92f110(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b82cf3e60aeb1e238f86f39b882ac93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e50f2f875757fc07d8b771b514df9a7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 104, 104], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62d800e6639e69fd1d31285c9a2bf5ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 184, 184], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 184, 184], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2930dc8d8f3add326f21394f6d894dde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
            ]


    class TestPrimitiveOp_06c0195f8c931bbf45800995eeaa48bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36165b228171aef746c929a38eee5c91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([784, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3136, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcbc36f339eceee99373ecf3ffdc2437(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([196, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([784, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3136, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4daa82ef86e98f3f0fc7a4b8499cdb5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([196, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([784, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3136, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0981cb4f48b75a57e1f92e789bdbc72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84fffd876e986fbe041708cfe62e24be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84fffd876e986fbe041708cfe62e24be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e69d67bd738a6dac320f413cd6b0a59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14a6e05f8b9bfbadf0b3979d861f18ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08131ee501a25c57c36f70c2062f7c92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08131ee501a25c57c36f70c2062f7c92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10fde12a6c621ab846209630c945bbe2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88676480c07e3b9b5e5d3bb35d957245(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6ee777da1f4d8de1fc653d49ec87003(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dc69633ef838b624f2dc23bc783f6f6
        def get_inputs(self):
            return [
                paddle.uniform([65280, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([16320, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4080, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1020, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_350c8edda8909054ecbb82044fcb22d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 65280, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16320, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4080, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1020, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7c23823e7c2732a012af79ac2dc0936(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 65280, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16320, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4080, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1020, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8aa0baec31fd4a9b706cb11eab8f58c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1009d3c9e400dc4d67f906a2c59f4a4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_dac7aa41654afb2cef1da307839c1674(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 200, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d44c59394b68087235efdd2ae7e543d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
            ]


    class TestPrimitiveOp_cff3bb13445432acde7ce3c98c749ac1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_8b261facfda4d39a8411366f5df5831f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([113061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b54beaad36878f6fbe33a902fdbc593e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb6657a136a14f583611b387d57e00f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cff3bb13445432acde7ce3c98c749ac1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_131c82038bb6f0355b489519f5976b52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc3d6ba772aba40377797242c2c054d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_066ded59fd3d558e40f24c13a55abc29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 512, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 160, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_703a1028a6969ba190f166ca310c655d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3fa29d4d75619f3cd4894a7a3e048626(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22b3e041679f5cfbeb59aa23f03a09a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc8d5c585901bdad5cbd5d68a2f5d60c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7596eb62fe7a9f95d8c13677d472504f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67bed2ecbf25b48f3c487bbdb6a9e4b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22a5753c61b4295f0f6bf67febeb96e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 232, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 232, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_072f08e60008f3d84dda8a6451fc7e94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5f88e839eb6d281035b18ff4daded60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            ]


    class TestPrimitiveOp_72be0766d1ec6db302397a32bfca7fed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94a9274e7a48563752533534b4ecaa4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([123783, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1268d15f43835604094ae512047fd84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 200, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_926e07052c6493355b6361030104c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_5373e9e13a9525dbc4a44cf84606954c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc7b5a32b4f842bef550d252ae27b8a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc3d6ba772aba40377797242c2c054d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ebc15059cd2e3329393fb9242f23612(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1eec25ec0cc270afa62f00ed11b1b15d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([185691, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_113d0a7238bb2d2ab820ca8e9e3859b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1dde18f45a2a20f0f44bb42bf0ff413(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 14, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 14, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10386854f9f57b07fc4bac22422dc0af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 42, 42], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a6d7982a15d190708f672b04317e168(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b286f2ddf2ae56cdeceebff61fbb446(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc3d6ba772aba40377797242c2c054d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_04fd00b5e7da8c1ed902b3a93477b175(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
            input_1 = 1
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_36c7a7cf3959badd153462377599eb3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04fd00b5e7da8c1ed902b3a93477b175
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 144, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69a59e3c94730251aecaf21d1c5f60cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9eb83f7be654a4f4ca222ad59bc6bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f099c6d40a62c39e6e5d3d3d69e08b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 100, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 100, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1009d3c9e400dc4d67f906a2c59f4a4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([5, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_f5e4965205e061c7db16e05d6556087e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b26eff381010e9d3dbc0e33c70f339b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc6299b330ebf3d40eca8a4556f8f2c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3166e46ed71a455b07a05e9d3d7f77d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29b9fa81183a10c24fc18b52c3142521
        def get_inputs(self):
            return [
                paddle.to_tensor([[4], [7], [8]], dtype='int64').reshape([3, 1]),
            ]


    class TestPrimitiveOp_e4bf4329285749c98b247b3989d7b2eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_efbbca4f20284a3b4996bcc2a704716c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 9, 9], dtype='int32').reshape([3]),
            ]


    class TestPrimitiveOp_659f43e482858296a61f52f77abc9894(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05fa05daa6c838ed81a83ab37a187aed
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_fce6a30e9e5e700a195534986806c162(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be9eb669746fd9491332d4aaee25f5ee
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06292838603258133, 0.30279308557510376, 0.39033249020576477], dtype='float32').reshape([3]),
            ]


    class TestPrimitiveOp_b4e6dfbb83c81de5e98e92132cfc98aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 512, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1ed2269484b12193c05a75314bfddfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc3d6ba772aba40377797242c2c054d4
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bed0b408f2530799bbce2286bf92f110(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([153450, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9eb83f7be654a4f4ca222ad59bc6bc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9e84bbbd05596a98b9be0c5104488c02(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fbc0f0c4b1ae5d6dd4fec32e73f3aa4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e84bbbd05596a98b9be0c5104488c02
        def get_inputs(self):
            return [
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f5fa530469a0854f581cefeac4c3cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91f3e8cfc3dc1f76c687591215954494(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.050307195633649826, 0.24958865344524384, 0.12819220125675201, 0.15536218881607056], [0.31391292810440063, 0.12240537256002426, 0.3002643585205078, 0.22792692482471466]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_1ac02bf307e9c5114a33594b1193d6e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.37339529395103455, 0.30681005120277405, 0.14150168001651764, 0.0019667954184114933]], dtype='float32').reshape([1, 4]),
            ]


    class TestPrimitiveOp_912d1e70b9821b23eaf10bdcc0e9608b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08483082801103592, 0.046015478670597076, 0.15802784264087677, 0.42296135425567627], [0.37927430868148804, 0.43439051508903503, 0.4780726134777069, 0.20804838836193085]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_e9bfa183e31798fc03f9d53f370de036(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e50ce9a3bb6f5c4f25fa24dfce50c59a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 4624, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1156, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 289, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e90ae374b3d80d508065d3103454cd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 4624, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1156, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 289, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_573a7ce37020be0d5682b46a1cdeb678(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 4624, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1156, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 289, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cae6e9a660f34bdc1e402cbc218fda1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([4624, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1156, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([289, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45b3dce5ab5ba23cfd6db22bcd704000(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([4624, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1156, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([289, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3ee04f2b25f862854073be798234eed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cae6e9a660f34bdc1e402cbc218fda1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([4624, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1156, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([289, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3738794ca2dbd9c33d4c55ab388aa621(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aeed443a0cef14b90625b6a17c4c06cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15910319983959198, 0.37697386741638184, 0.07013243436813354, 0.14398092031478882], [0.3346053659915924, 0.4726791977882385, 0.03578709438443184, 0.40225374698638916]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_a3242ac7d5e05d6a96595b6530ca70c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.015704719349741936, 0.1316700279712677, 0.0756133571267128, 0.4947950839996338]], dtype='float32').reshape([1, 4]),
            ]


    class TestPrimitiveOp_19545642fb3a8e976dc673d2b6d243ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3335762619972229, 0.255480021238327, 0.28426453471183777, 0.41945880651474], [0.26453742384910583, 0.30485397577285767, 0.11422444134950638, 0.10182178020477295]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_a0b951fee4fac17f20e9f1d7c3bf42d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dc69633ef838b624f2dc23bc783f6f6
        def get_inputs(self):
            return [
                paddle.uniform([84864, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([21216, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5304, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1326, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([351, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd846d6dd4551d59311149db20e660ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 84864, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21216, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5304, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1326, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 351, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9489fe61b5c5a941aa9d9ede990bf8d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 84864, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21216, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5304, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1326, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 351, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe312f3264046e4b951a8028052bb725(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_49f6cac1efebf69bc20b023aef52ac51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_391a7c9492aa8289e75c65cd977d9e59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e03fdd4df2e6978a2aa5d810a24520a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f476aa231bf54a95952cd42a5457609(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51436c8b15ec8341a7167d2213ec7fee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a63d36e4c21637f6f4105b4637c8178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c07c2cbc7e23158bb1c552388ef59ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b26eff381010e9d3dbc0e33c70f339b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2efcd9a6a5994a82745e3d2c86eb7a5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed9e5ed03eb322191c2a51a0bede6911(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc3d6ba772aba40377797242c2c054d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5f88e839eb6d281035b18ff4daded60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            ]


    class TestPrimitiveOp_b3c49d03dd35087a2764180bb4a6d483(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_796c7ce66f63beef7121dd47a2030215(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea4755e42ca7e1594e982a44a869380e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37eecf83334d0b05052f1b3d8119c6a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ad450993114745f791e451f23b0d4af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_985ced31834c8cde77a9f99dc1a09625(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad604651d7c2875f187533dfcb3f3a1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12f6631af91f636265712dbf46fc81e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29b9fa81183a10c24fc18b52c3142521
        def get_inputs(self):
            return [
                paddle.to_tensor([[4], [7]], dtype='int64').reshape([2, 1]),
            ]


    class TestPrimitiveOp_178f7495d58364f50dbcad58a954c2b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_688332b39c6b124768473e4ec975e07d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 2], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_ed5fcadcb2bcb4a336b2565af2ccd1ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05fa05daa6c838ed81a83ab37a187aed
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_dba3b02e6e14fcf6968e1277b994f5a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be9eb669746fd9491332d4aaee25f5ee
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06586539000272751, 0.08793290704488754], dtype='float32').reshape([2]),
            ]


    class TestPrimitiveOp_b5a1ee389ee7e4fde6d9c6b44e48a82f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebcca779e28a16eb25c50edb6a417e73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            ]


    class TestPrimitiveOp_4c54f4d654dd1750fd613b06a5a32be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3ee04f2b25f862854073be798234eed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b207547ccceedf5026010f04b6e27536(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6dcac3f199c6692b1ec3b33123ecd39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90921eaea4d82de2c4f9271ad8876a65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_878bf3628f706664a89740d9055781c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc3d6ba772aba40377797242c2c054d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34fb740256cd5266386d5a040913c667(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb0d258ca0e89ef345e0f2dbdd6e13b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d462a02270c2baa508cfcc78564bb95f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad7da2dcac1f5a2934a93d55d509edf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dbfcff39bb5bd9077837078382dc6ca6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_094c26ab155bbff06c88308ee68f0f49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 92, 92], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81c245d0066a33512c3e54b4f2094e91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43efc822ed6a864e17bbbd9c04aa4b5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8004b31845c6aeb0d749866c7d435896(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dc69633ef838b624f2dc23bc783f6f6
        def get_inputs(self):
            return [
                paddle.uniform([139392, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([34848, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([8712, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2178, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([561, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9c20df2b18207eac4e2a5ca08d68cbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 139392, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 34848, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8712, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2178, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 561, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02b50954cb4f283f0f1d6d70bce6e617(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 139392, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 34848, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8712, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2178, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 561, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f89e11afac043fb4e3fd045ba4ce53be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_543b4125ed9558644a0e3b19d007cc5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09dbe901a54cd57fa2c471870b04179f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_4930120b340c048b5e779bfb4228e660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_576d0f4b0cc4c0987bf4031d8b112e0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e66513c47702db069bc11704b05fd587(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fcaa0d2ab697e57ef2552383dab1052d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1924fbd604fb367cba9be9f22d8510e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b2a9e1a310bf4cd258faeef52ec86eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80a7c0d8632d388263ab6994bde0ef15
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4, 1, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43f41e277fd78ed0f59449d715dd7d53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 46, 46], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db417aceaa16e5479585629d16b8f124(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dc69633ef838b624f2dc23bc783f6f6
        def get_inputs(self):
            return [
                paddle.uniform([15200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2274236d6d79a3246d7eb843d9ede4fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dc69633ef838b624f2dc23bc783f6f6
        def get_inputs(self):
            return [
                paddle.uniform([15200, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b57a866c33b966b878d1c96a59fe2351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dc69633ef838b624f2dc23bc783f6f6
        def get_inputs(self):
            return [
                paddle.uniform([15200, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f437b74811e03a453ef693a5ca4ed3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4555ceef6bb4aee5364a8b1d69750c87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c93def0722a85d4c9a7c4d7a742c7139(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dc69633ef838b624f2dc23bc783f6f6
        def get_inputs(self):
            return [
                paddle.uniform([163200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([40800, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([10200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2550, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([663, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b73d7cc05e9f5d08d3389f56991cd019(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 163200, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40800, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 10200, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2550, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 663, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bd5c71070b2feb6851fb28044c2fe75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 163200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40800, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 10200, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2550, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 663, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfc26f9f2fb43310e7bb404ae2b220c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d04540c13157757630a3380fdabe8428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 84, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb0d258ca0e89ef345e0f2dbdd6e13b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d462a02270c2baa508cfcc78564bb95f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad7da2dcac1f5a2934a93d55d509edf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4734ce31d43c9cb46ee820a48f86fa30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0929a44e57036e7e527ff59bfee1599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_107db5c5d0eb41cd9bb81e273f70e173(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d53b8b87f9bc573de0d128526842e95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_164adfad06daeddb2b07f25c1f18ad09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_725540385391b431c66103dffd16a95c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0532ad905f5ca95296e7a34bee2cd09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_405e475e725e02bd87c9d35d0b8454b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dc69633ef838b624f2dc23bc783f6f6
        def get_inputs(self):
            return [
                paddle.uniform([92928, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([23232, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5808, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1452, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([363, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_690ab43e77028e3f72f2ec48706ec738(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 92928, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 23232, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5808, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1452, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 363, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1af1bc20e1aa9da6848285a8a335ba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 92928, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 23232, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5808, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1452, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 363, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8fc2a6fc927723874768cc238c8df9d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce3fc482a7b98eb99b64ae7092653ced(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd88d5cc45b76fe8e5591307e29259b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9b9b95cf4c8387a1c38f4872b17d3be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ca1911ead4daa7197c300fd2bf878ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd217c1f3710c877bb7c40897ef98c01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8eb1c3bc24e21f64d2b119df16f5c3dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 150, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5f88e839eb6d281035b18ff4daded60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            ]


    class TestPrimitiveOp_b27ed6e19cdaac38593b2302476affac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6912ee2adeec91822688b63c5214800d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 116, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 116, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f135f3be14a638d02300445feaa8e46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b668d235f7eedf6d53a33e3a424654b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2eb3b35d6166cc0e15d8fcd5894993ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69aa063539d88d13f65ef26d61127020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e5478e11e4ef61f5f473a43716b2b3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5373e9e13a9525dbc4a44cf84606954c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_310d7c57b0bbc028391a9f35d945f8a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 120, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 36, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e8f5c01f959e00c320c7283b690d981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36d3cea0fe7ddf76b5b8572e1637a193(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dc69633ef838b624f2dc23bc783f6f6
        def get_inputs(self):
            return [
                paddle.uniform([154560, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([38640, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([9660, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2415, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([648, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01348bdf7e68623b9cab051174dffa11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 154560, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 38640, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9660, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2415, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 648, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a065556f09b1500a230b2a032607201b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 154560, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 38640, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9660, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2415, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 648, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84f491f154fc645c15d3983011bdcd8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84f491f154fc645c15d3983011bdcd8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78fa064419713f2b59aca40b0881fe0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55425ba3888dca3d6b66400fbcb5b152(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_864cd9d66f7a7cbca9d151843a4106bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_864cd9d66f7a7cbca9d151843a4106bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a16637552e58e14c59d6adab40d51c45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_192abee4332d161bf045d255f841a8bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b94e6c5768630ef8202353d8763f976(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e129c953d9fbe51717221bb8768f9361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e87959ec12b4b84b2b8766c0d176f33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 224, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 128, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 128, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 128, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 128, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b78014c0f62c0a690bd2ef5734a5393(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca00c57df3314141f24a750357608d55
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_f6a6531be8d5a875f3b37f228fdb22bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1328401f728849a715eaaaf50b1b739f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
                paddle.randint(low=0, high=3, shape=[1, 500, 1], dtype='int64'),
            ]


    class TestPrimitiveOp_4521833aa42f6500af247b0a008490bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 256, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 60, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4372084ebfd9ac26af20af2f83be987(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc3d6ba772aba40377797242c2c054d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff6d9c177af939e542c6928e3a61bfdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f579ffdd1f269b7810026e3cf3962e14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1a8a94174e762fbe950c17cae2824d09(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48, arg_0_49, arg_0_50, arg_0_51, arg_0_52, arg_0_53, arg_0_54, arg_0_55, arg_0_56, arg_0_57, arg_0_58, arg_0_59, arg_0_60, arg_0_61, arg_0_62, arg_0_63, arg_0_64, arg_0_65, arg_0_66, arg_0_67, arg_0_68, arg_0_69, arg_0_70, arg_0_71, arg_0_72, arg_0_73, arg_0_74, arg_0_75, arg_0_76, arg_0_77, arg_0_78, arg_0_79, arg_0_80, arg_0_81, arg_0_82, arg_0_83, arg_0_84, arg_0_85, arg_0_86, arg_0_87, arg_0_88, arg_0_89, arg_0_90, arg_0_91, arg_0_92, arg_0_93, arg_0_94, arg_0_95, arg_0_96, arg_0_97, arg_0_98, arg_0_99, arg_0_100, arg_0_101, arg_0_102, arg_0_103, arg_0_104, arg_0_105, arg_0_106, arg_0_107, arg_0_108, arg_0_109, arg_0_110, arg_0_111, arg_0_112, arg_0_113, arg_0_114, arg_0_115, arg_0_116, arg_0_117, arg_0_118, arg_0_119, arg_0_120, arg_0_121, arg_0_122, arg_0_123, arg_0_124, arg_0_125, arg_0_126, arg_0_127, arg_0_128, arg_0_129, arg_0_130, arg_0_131, arg_0_132, arg_0_133, arg_0_134, arg_0_135, arg_0_136, arg_0_137, arg_0_138, arg_0_139, arg_0_140, arg_0_141, arg_0_142, arg_0_143, arg_0_144, arg_0_145, arg_0_146, arg_0_147, arg_0_148, arg_0_149, arg_0_150, arg_0_151, arg_0_152, arg_0_153, arg_0_154, arg_0_155, arg_0_156, arg_0_157, arg_0_158, arg_0_159, arg_0_160, arg_0_161, arg_0_162, arg_0_163, arg_0_164, arg_0_165, arg_0_166, arg_0_167, arg_0_168, arg_0_169, arg_0_170, arg_0_171, arg_0_172, arg_0_173, arg_0_174, arg_0_175, arg_0_176, arg_0_177, arg_0_178, arg_0_179, arg_0_180, arg_0_181, arg_0_182, arg_0_183, arg_0_184, arg_0_185, arg_0_186, arg_0_187, arg_0_188, arg_0_189, arg_0_190, arg_0_191, arg_0_192, arg_0_193, arg_0_194, arg_0_195):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12, arg_0_13, arg_0_14, arg_0_15, arg_0_16, arg_0_17, arg_0_18, arg_0_19, arg_0_20, arg_0_21, arg_0_22, arg_0_23, arg_0_24, arg_0_25, arg_0_26, arg_0_27, arg_0_28, arg_0_29, arg_0_30, arg_0_31, arg_0_32, arg_0_33, arg_0_34, arg_0_35, arg_0_36, arg_0_37, arg_0_38, arg_0_39, arg_0_40, arg_0_41, arg_0_42, arg_0_43, arg_0_44, arg_0_45, arg_0_46, arg_0_47, arg_0_48, arg_0_49, arg_0_50, arg_0_51, arg_0_52, arg_0_53, arg_0_54, arg_0_55, arg_0_56, arg_0_57, arg_0_58, arg_0_59, arg_0_60, arg_0_61, arg_0_62, arg_0_63, arg_0_64, arg_0_65, arg_0_66, arg_0_67, arg_0_68, arg_0_69, arg_0_70, arg_0_71, arg_0_72, arg_0_73, arg_0_74, arg_0_75, arg_0_76, arg_0_77, arg_0_78, arg_0_79, arg_0_80, arg_0_81, arg_0_82, arg_0_83, arg_0_84, arg_0_85, arg_0_86, arg_0_87, arg_0_88, arg_0_89, arg_0_90, arg_0_91, arg_0_92, arg_0_93, arg_0_94, arg_0_95, arg_0_96, arg_0_97, arg_0_98, arg_0_99, arg_0_100, arg_0_101, arg_0_102, arg_0_103, arg_0_104, arg_0_105, arg_0_106, arg_0_107, arg_0_108, arg_0_109, arg_0_110, arg_0_111, arg_0_112, arg_0_113, arg_0_114, arg_0_115, arg_0_116, arg_0_117, arg_0_118, arg_0_119, arg_0_120, arg_0_121, arg_0_122, arg_0_123, arg_0_124, arg_0_125, arg_0_126, arg_0_127, arg_0_128, arg_0_129, arg_0_130, arg_0_131, arg_0_132, arg_0_133, arg_0_134, arg_0_135, arg_0_136, arg_0_137, arg_0_138, arg_0_139, arg_0_140, arg_0_141, arg_0_142, arg_0_143, arg_0_144, arg_0_145, arg_0_146, arg_0_147, arg_0_148, arg_0_149, arg_0_150, arg_0_151, arg_0_152, arg_0_153, arg_0_154, arg_0_155, arg_0_156, arg_0_157, arg_0_158, arg_0_159, arg_0_160, arg_0_161, arg_0_162, arg_0_163, arg_0_164, arg_0_165, arg_0_166, arg_0_167, arg_0_168, arg_0_169, arg_0_170, arg_0_171, arg_0_172, arg_0_173, arg_0_174, arg_0_175, arg_0_176, arg_0_177, arg_0_178, arg_0_179, arg_0_180, arg_0_181, arg_0_182, arg_0_183, arg_0_184, arg_0_185, arg_0_186, arg_0_187, arg_0_188, arg_0_189, arg_0_190, arg_0_191, arg_0_192, arg_0_193, arg_0_194, arg_0_195]
            input_1 = 0
            return paddle._C_ops.concat(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3246055e85f0074ac80925f797b4199b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a8a94174e762fbe950c17cae2824d09
        def get_inputs(self):
            return [
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6dd564fc4701c2f1e14ee3caaee2c2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 200, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f35595e915ee652c92bcd73ce4f849c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4594e9fa9046f61566d33ba89135fac0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dc69633ef838b624f2dc23bc783f6f6
        def get_inputs(self):
            return [
                paddle.uniform([165888, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([41472, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([10368, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2592, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([648, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0bad33f06a5372a751cfba3fa155643(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 165888, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 41472, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 10368, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2592, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 648, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9401fec327d71b7e2c8743bc4046fe1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f91207dcf5373d03e0d4bae9a0e638
        def get_inputs(self):
            return [
                paddle.uniform([1, 165888, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 41472, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 10368, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2592, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 648, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70468b99a2e60a5c01db2c12f9b28967(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 5184, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1296, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 324, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aef8e1e9d8bf6c57d0393c12c79e392f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 5184, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1296, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 324, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03ab6a91ec6481d5bf96159639c5c485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 5184, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1296, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 324, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0a6f393d874d5b59f6579e4b0fc58af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([5184, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1296, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([324, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5637a5a1caa89e74be340b06dae5d67a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([5184, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1296, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([324, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ef8faf3f818611c34655e19bd939f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0a6f393d874d5b59f6579e4b0fc58af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3690a95e8d928803eb155a64110edfc
        def get_inputs(self):
            return [
                paddle.uniform([5184, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1296, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([324, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d886314fa1cdf3122be7a8cf534c4b13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc8987d0fa81c1bfdd44ed683430ae39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c47325a89c1df5419e84c1de33c21731
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
            ]


    class TestPrimitiveOp_1015e47a7883f59812b13b36c4917d22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e13b3cace1eab17769e3d996ba8d5aba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([220968, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9767a5ef7e9301ab7140682cbbda96a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f7d3be4d352571e6b0b689556243d71
        def get_inputs(self):
            return [
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5507dbdb100ef98aee03804951311fd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e84bbbd05596a98b9be0c5104488c02
        def get_inputs(self):
            return [
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27cf214162415ce24542e8bdd57d6ebe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_442b9549cdf4a3b2707607968f296f23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6fc16552947b136b44f18e835102be9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80a7c0d8632d388263ab6994bde0ef15
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4, 1, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91c94c58912f397f67e61c6c1dc565b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4e70a9c135795cd568337f5637ba86e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec7aec00e621c872d839ead654016713(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63efec9f0888d312f930d07c78e9e9fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80a7c0d8632d388263ab6994bde0ef15
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4, 1, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_deda542b498eb303c3a11ebc6dbcd8cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 58, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 58, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af6b69d335adf9794f556a076c8a729c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3b0cecf503832cde54979db6d853720(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e35e1799178626f7263cfaba0c944d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a5045cb4a40923ed322b0780394b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68ff6fa7edcd06bc9242df895016d196(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8367bddd16447487644cf634ef318ea
        def get_inputs(self):
            return [
                paddle.uniform([185658, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4734ce31d43c9cb46ee820a48f86fa30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0929a44e57036e7e527ff59bfee1599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f215a0ad02d0707d2dfa4f16bdfe18
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0cc2f0d23c249eee4a7dcb537d51fb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f738c1d8584e892956793930d78a2b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5437b5b721fa7493777677542c2c152(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80a7c0d8632d388263ab6994bde0ef15
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4, 1, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41a443ae64993eaa191dbbfc44f2929a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70a0ec37af6347f478f430b9baa63c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 512, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cff3bb13445432acde7ce3c98c749ac1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([7, 256, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 7, 7]),
            ]


    class TestPrimitiveOp_94b20e15caad3eb181feea47ae2e72d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2264301399ba23f3e3c3d6850b54a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 256, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 36, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_274a4a3f403c685000e3e4a100c41d86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_899c6f506ffb34c52f62e183d8304069
        def get_inputs(self):
            return [
                paddle.uniform([6, 256, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
                paddle.to_tensor([], dtype='float32').reshape([0, 256, 14, 14]),
            ]


    

if __name__ == '__main__':
    unittest.main()