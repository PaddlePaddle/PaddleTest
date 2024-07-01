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
    class PrimitiveOp_f0151b39dee5a1b576acbaeceac3d0a0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_968b1fab7d6dc74add8b4fb345b46f57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0151b39dee5a1b576acbaeceac3d0a0
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([300], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_eefe8c83939a10a0ee1bba47e728b139(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6439baa7dd5c0d2c05914079586736e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eefe8c83939a10a0ee1bba47e728b139
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_3fd97a506005a7f14699dd6857d2a96c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_782545b582a57e83b21e3b3c15953c4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fd97a506005a7f14699dd6857d2a96c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_ba82c904c45357111484d72dc53a5516(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8a972712e7b4fbb7d0bbfaa21b685325(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba82c904c45357111484d72dc53a5516
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_81882b7a84968b2418283b76cdbbe27d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d29d80350d9fd59c32aa5e03c856633e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81882b7a84968b2418283b76cdbbe27d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_b6bbf425d31b170c975112cae854aadd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_12ac46002d52c3ac2660206a6942b923(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6bbf425d31b170c975112cae854aadd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_0ecc303f8ddfb9bd09e4b17cc7bee71f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5e879725882cd5ae0e67fac0a7839cb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ecc303f8ddfb9bd09e4b17cc7bee71f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_7af88b4e4a2a446f8915c0dbb62f30bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_75504c59c98f3d5ecb38dbdf0b120ab3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7af88b4e4a2a446f8915c0dbb62f30bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_40efff081eada4fd48d582ea4d068596(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 168, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bde3adcfef9cb2d9c694f6aa98fcbadc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40efff081eada4fd48d582ea4d068596
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.13313806056976318, 0.4626670777797699, 0.46212807297706604, 0.25356119871139526], [0.12604497373104095, 0.47608232498168945, 0.47727617621421814, 0.3902834355831146]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_43c4dde1e932943e9cd2e27cb58032dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 84, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a6a8db8c04e259950e2d5e7bd1a7c990(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43c4dde1e932943e9cd2e27cb58032dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_f12ae4e1d8653d9d46d5009329c072cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 42, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4eb2e00329e40460299ae3ac9f9f25ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f12ae4e1d8653d9d46d5009329c072cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_42a3a551539068567debf69f79425812(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 21, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce2f5d4cbc404b891bd416e849d2566f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42a3a551539068567debf69f79425812
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_14b13389bfa5cd01718765afa3470dd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0151b39dee5a1b576acbaeceac3d0a0
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([100], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0cb84f53ee15a6263c2b478b4e820a24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eefe8c83939a10a0ee1bba47e728b139
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bced4277c70f37187041e1c3937d4885(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fd97a506005a7f14699dd6857d2a96c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e0adbbc8bed696fa8c4676e50c58bde9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba82c904c45357111484d72dc53a5516
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cda93d9649d0d7ff8c17a4bdeadb01a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81882b7a84968b2418283b76cdbbe27d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.08681345731019974, 0.18166474997997284, 0.365119993686676, 0.0067316764034330845], [0.10514085739850998, 0.4515710771083832, 0.0026658447459340096, 0.3969408869743347]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_071d7f42eb7ad0d8226adcd5c073406b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6bbf425d31b170c975112cae854aadd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 68, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_fb9366db041e11be6e9f7c797f633bce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ecc303f8ddfb9bd09e4b17cc7bee71f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 34, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7abe953b67fa414753f4c39a8032f8b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7af88b4e4a2a446f8915c0dbb62f30bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_d472de780bfc53fb789d4fb20132b4ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 200, 304], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a5fcb400f82bfe77f4075480d6d60214(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d472de780bfc53fb789d4fb20132b4ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.47624194622039795, 0.4252462089061737, 0.4653246998786926, 0.4718214273452759], [0.26620176434516907, 0.39858391880989075, 0.11420700699090958, 0.23182038962841034]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_cc9dfabc386a7f7e23db0777eb113b30(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 100, 152], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_804b2b4730b4986b21868a3e747087a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc9dfabc386a7f7e23db0777eb113b30
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_220e70a43da3b1bb8823f2b877b76f44(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 50, 76], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6b8c2e1069d0a135f8ee811ee244541d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_220e70a43da3b1bb8823f2b877b76f44
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_d060f63cb10887a39cf98ac9278f0f77(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 25, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e96c864c49c141225ea91a876d4bb0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d060f63cb10887a39cf98ac9278f0f77
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d29d80350d9fd59c32aa5e03c856633e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81882b7a84968b2418283b76cdbbe27d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_12ac46002d52c3ac2660206a6942b923(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6bbf425d31b170c975112cae854aadd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5e879725882cd5ae0e67fac0a7839cb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ecc303f8ddfb9bd09e4b17cc7bee71f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_75504c59c98f3d5ecb38dbdf0b120ab3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7af88b4e4a2a446f8915c0dbb62f30bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_36d7d72dd6131ef332708ef0abdfd06b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81882b7a84968b2418283b76cdbbe27d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.49739229679107666, 0.3351207375526428, 0.4484364688396454, 0.12505020201206207], [0.030114924535155296, 0.19822652637958527, 0.2196095734834671, 0.40188801288604736], [0.37833085656166077, 0.005445803515613079, 0.4010592997074127, 0.42676693201065063], [0.35176393389701843, 0.16488398611545563, 0.10220362991094589, 0.2679820656776428], [0.20115259289741516, 0.23507601022720337, 0.2623601257801056, 0.21225251257419586], [0.3057858347892761, 0.18095499277114868, 0.42079871892929077, 0.12342901527881622], [0.1707804948091507, 0.18216678500175476, 0.05291115120053291, 0.3348720669746399]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b2659266fa50ac4c9f62f8fb9a0a8479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6bbf425d31b170c975112cae854aadd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_df1251e2b7856f9153a9c11d25c6874c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ecc303f8ddfb9bd09e4b17cc7bee71f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_528e35e077b9ca3221d3d828dfea6319(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7af88b4e4a2a446f8915c0dbb62f30bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_85641fc0a9c8a7b32d0053b7be1b007c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7011d8597a92c9b1e100052f461f03d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85641fc0a9c8a7b32d0053b7be1b007c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.12931282818317413, 0.48861533403396606, 0.17718838155269623, 0.12520994246006012], [0.07523564994335175, 0.4386076331138611, 0.3620389699935913, 0.07658398151397705], [0.14756430685520172, 0.07572465389966965, 0.26838070154190063, 0.37414035201072693], [0.3915621042251587, 0.19104118645191193, 0.01597001403570175, 0.05103715509176254], [0.22588086128234863, 0.0050764307379722595, 0.31528225541114807, 0.298330157995224], [0.177719846367836, 0.25171852111816406, 0.10758128017187119, 0.3024943470954895]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([6], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_51d402daab72be882d05e4f57bacef27(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f93ceebf84c04bb939c54e50dac39487(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51d402daab72be882d05e4f57bacef27
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_1d0365a47f1f4b9ab59d7f90970ba8fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_70aef9d54d49147910723c375db4209c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d0365a47f1f4b9ab59d7f90970ba8fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_543794d4d0417e476e2bac89c6ecbbdb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_016739b32e0f82e3a220245ac889b862(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_543794d4d0417e476e2bac89c6ecbbdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_69a9e5435dba619c646d943bc67665a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81882b7a84968b2418283b76cdbbe27d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.1701001077890396, 0.014843899756669998, 0.09852124750614166, 0.3393995761871338], [0.11748797446489334, 0.2589859068393707, 0.31148210167884827, 0.10683373361825943], [0.3116267919540405, 0.4089503586292267, 0.36851558089256287, 0.13546034693717957]], dtype='float32').reshape([3, 4]),
                paddle.to_tensor([3], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b8a3a3bf7484fe66b4069e739f3b1765(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6bbf425d31b170c975112cae854aadd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 136], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9b62a0f36ddfa1b39bc8d3d887c86d75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ecc303f8ddfb9bd09e4b17cc7bee71f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 68], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4d127a894e4fe9a127c743014be97d6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7af88b4e4a2a446f8915c0dbb62f30bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ea590ccb08ae6982b6707b8ae0be9025(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d472de780bfc53fb789d4fb20132b4ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.049569591879844666, 0.30775371193885803, 0.3094034790992737, 0.13927102088928223], [0.24713994562625885, 0.37523457407951355, 0.024224577471613884, 0.3512221872806549]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_804b2b4730b4986b21868a3e747087a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc9dfabc386a7f7e23db0777eb113b30
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6b8c2e1069d0a135f8ee811ee244541d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_220e70a43da3b1bb8823f2b877b76f44
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9e96c864c49c141225ea91a876d4bb0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d060f63cb10887a39cf98ac9278f0f77
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_39058bff81934cedcc42132f61450f6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85641fc0a9c8a7b32d0053b7be1b007c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.03567306697368622, 0.41631054878234863, 0.14406552910804749, 0.48773443698883057]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9f12d4fa7bf610eb1da7cb2914f8fdfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51d402daab72be882d05e4f57bacef27
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_21b731e6f3997e40ea1b443a13600c2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d0365a47f1f4b9ab59d7f90970ba8fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c864bcdca29d412ac7feb3649df87e01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_543794d4d0417e476e2bac89c6ecbbdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a6950f540f89083bc74ea4b93cffe39a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81882b7a84968b2418283b76cdbbe27d
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.33109039068222046, 0.08781209588050842, 0.023874972015619278, 0.4068280756473541], [0.27392226457595825, 0.46916234493255615, 0.3789480924606323, 0.32833385467529297], [0.3113239109516144, 0.34468361735343933, 0.4564971327781677, 0.373810350894928], [0.20377878844738007, 0.48164132237434387, 0.3123581111431122, 0.20703266561031342], [0.44531503319740295, 0.4649951756000519, 0.18325623869895935, 0.19394664466381073], [0.322318434715271, 0.3634970784187317, 0.12563112378120422, 0.49102601408958435], [0.2974511981010437, 0.2236485332250595, 0.2726476192474365, 0.3250909149646759]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a628e6d80ae845ae552a2e7fa4ad12de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6bbf425d31b170c975112cae854aadd
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 68, 104], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_97648d77abb90145f9e9e2b60f13d81c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ecc303f8ddfb9bd09e4b17cc7bee71f
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 34, 52], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b797d53731a8cad88c189ad5f8d117e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7af88b4e4a2a446f8915c0dbb62f30bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_82c027a1d9009b9df66f1ecf20ccc953(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85641fc0a9c8a7b32d0053b7be1b007c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4272349774837494, 0.47877711057662964, 0.08770851790904999, 0.08134938776493073]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f93ceebf84c04bb939c54e50dac39487(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51d402daab72be882d05e4f57bacef27
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_70aef9d54d49147910723c375db4209c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d0365a47f1f4b9ab59d7f90970ba8fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_016739b32e0f82e3a220245ac889b862(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_543794d4d0417e476e2bac89c6ecbbdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_aabf925cbb4c1d4abbb127ee96dcd2b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81882b7a84968b2418283b76cdbbe27d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.2777930796146393, 0.02657657489180565, 0.47445645928382874, 0.20409588515758514], [0.1387108713388443, 0.39539310336112976, 0.1251879781484604, 0.12581704556941986], [0.0013369943480938673, 0.06631842255592346, 0.38734182715415955, 0.07199130952358246], [0.4746476411819458, 0.005258830729871988, 0.14967621862888336, 0.14246855676174164], [0.15963207185268402, 0.3332064747810364, 0.09021896123886108, 0.44095081090927124]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([5], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6645c6167f784b502e2cf26cf40f3b19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6bbf425d31b170c975112cae854aadd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_3523dd2abf9ebdf3fcbf08892b155325(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ecc303f8ddfb9bd09e4b17cc7bee71f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a83899a60f9d3aac6beace12a1b2fd6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7af88b4e4a2a446f8915c0dbb62f30bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6ca1bf52e1efedb887d2bda2973a0cc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81882b7a84968b2418283b76cdbbe27d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.36710721254348755, 0.2733869254589081, 0.12741735577583313, 0.09983520209789276], [0.1689828783273697, 0.12775754928588867, 0.4929034411907196, 0.024316783994436264], [0.28232342004776, 0.3154520094394684, 0.34447357058525085, 0.1899302452802658], [0.3894905745983124, 0.21266524493694305, 0.15552382171154022, 0.2049335241317749], [0.14210514724254608, 0.42684876918792725, 0.16730007529258728, 0.03494313359260559], [0.36946699023246765, 0.25839653611183167, 0.40660321712493896, 0.049302250146865845], [0.4219869375228882, 0.046291451901197433, 0.45390188694000244, 0.2299237996339798]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8cf32724159a136b56691ea6c925bfa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6bbf425d31b170c975112cae854aadd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a87bbe6f9c0968c05f91b7a69992f19f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ecc303f8ddfb9bd09e4b17cc7bee71f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ed45fe8b53310a8bc4ca8c11ca76950c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7af88b4e4a2a446f8915c0dbb62f30bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e1301c36bcc3b8f6e0509f19de5e034c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81882b7a84968b2418283b76cdbbe27d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.04564209654927254, 0.4040566384792328, 0.4310336410999298, 0.19761954247951508], [0.06333372741937637, 0.015020381659269333, 0.26050060987472534, 0.4925742447376251], [0.004450720734894276, 0.09676632285118103, 0.15310807526111603, 0.2296193391084671], [0.3995705544948578, 0.20686708390712738, 0.16882404685020447, 0.41516441106796265], [0.10278435796499252, 0.2873406708240509, 0.33401980996131897, 0.45226940512657166], [0.16829141974449158, 0.3355092406272888, 0.43085777759552, 0.3128385841846466], [0.39289021492004395, 0.4308372735977173, 0.11628603935241699, 0.23280969262123108]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b2659266fa50ac4c9f62f8fb9a0a8479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6bbf425d31b170c975112cae854aadd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_df1251e2b7856f9153a9c11d25c6874c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ecc303f8ddfb9bd09e4b17cc7bee71f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_528e35e077b9ca3221d3d828dfea6319(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7af88b4e4a2a446f8915c0dbb62f30bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2dd36a459032fb730ffb1152534c7bbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85641fc0a9c8a7b32d0053b7be1b007c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.06026938185095787, 0.10351205617189407, 0.010360955260694027, 0.05767056718468666]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5f1c7b69da1443187a4314145df18c85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51d402daab72be882d05e4f57bacef27
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4d5e64918adb0dcc692607da607e76a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d0365a47f1f4b9ab59d7f90970ba8fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_06faa98b8d70222f62408892ac0f3e4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_543794d4d0417e476e2bac89c6ecbbdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_968b1fab7d6dc74add8b4fb345b46f57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0151b39dee5a1b576acbaeceac3d0a0
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([300], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6439baa7dd5c0d2c05914079586736e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eefe8c83939a10a0ee1bba47e728b139
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_782545b582a57e83b21e3b3c15953c4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fd97a506005a7f14699dd6857d2a96c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8a972712e7b4fbb7d0bbfaa21b685325(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba82c904c45357111484d72dc53a5516
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_44fc9d6f579c5072180e9037382072c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81882b7a84968b2418283b76cdbbe27d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.11586524546146393, 0.3802587389945984, 0.2655188739299774, 0.32558372616767883], [0.2183745801448822, 0.062354110181331635, 0.09305060654878616, 0.19626672565937042], [0.051588743925094604, 0.2933014929294586, 0.06933026015758514, 0.34952929615974426], [0.4719468057155609, 0.09559851884841919, 0.3501664698123932, 0.04593765735626221], [0.3326735496520996, 0.0424429252743721, 0.33680230379104614, 0.1775297373533249]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([5], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6645c6167f784b502e2cf26cf40f3b19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6bbf425d31b170c975112cae854aadd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_3523dd2abf9ebdf3fcbf08892b155325(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ecc303f8ddfb9bd09e4b17cc7bee71f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a83899a60f9d3aac6beace12a1b2fd6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7af88b4e4a2a446f8915c0dbb62f30bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d5a828781268e72042786f101bfcb43c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81882b7a84968b2418283b76cdbbe27d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3226352334022522, 0.1387707144021988, 0.13630889356136322, 0.3679457902908325]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f4e27e62b33813e391a8d3aa84c24f1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6bbf425d31b170c975112cae854aadd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8f5bf4fb44b2d637e03cc1335b5e6018(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ecc303f8ddfb9bd09e4b17cc7bee71f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b3cbad14996bd6dab4d5b40f17253b60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7af88b4e4a2a446f8915c0dbb62f30bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_388c9ad81e86b3083b4559e938bef7ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85641fc0a9c8a7b32d0053b7be1b007c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4692842364311218, 0.07899530231952667, 0.43798011541366577, 0.0650915876030922]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e28422e4f18889d022ec6c1272af3239(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51d402daab72be882d05e4f57bacef27
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d191b2f9938720291ac12b9f79ad520f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d0365a47f1f4b9ab59d7f90970ba8fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b93c8d0a8aaf39db862152ab40d76850(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_543794d4d0417e476e2bac89c6ecbbdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d1115bd9f0153403a925ea23393900a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85641fc0a9c8a7b32d0053b7be1b007c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5f1c7b69da1443187a4314145df18c85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51d402daab72be882d05e4f57bacef27
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4d5e64918adb0dcc692607da607e76a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d0365a47f1f4b9ab59d7f90970ba8fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_06faa98b8d70222f62408892ac0f3e4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_543794d4d0417e476e2bac89c6ecbbdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_14b13389bfa5cd01718765afa3470dd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0151b39dee5a1b576acbaeceac3d0a0
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([100], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0cb84f53ee15a6263c2b478b4e820a24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eefe8c83939a10a0ee1bba47e728b139
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bced4277c70f37187041e1c3937d4885(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fd97a506005a7f14699dd6857d2a96c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e0adbbc8bed696fa8c4676e50c58bde9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba82c904c45357111484d72dc53a5516
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_08c48b28c4f7dc25c68a5c0251e6c7cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85641fc0a9c8a7b32d0053b7be1b007c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.2400294840335846, 0.33962640166282654, 0.2767137289047241, 0.09159930795431137]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b95226fb2c555b521a452da3c6a12505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51d402daab72be882d05e4f57bacef27
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2f9023d253cffd1d2f6b45942617af40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d0365a47f1f4b9ab59d7f90970ba8fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1f42dfd36396fa06dd057d2c0b3c3a93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_543794d4d0417e476e2bac89c6ecbbdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8d49b380136f41736d416409da048b01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81882b7a84968b2418283b76cdbbe27d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.13706611096858978, 0.08502600342035294, 0.44341975450515747, 0.3082224726676941], [0.009993945248425007, 0.3372018337249756, 0.11831139028072357, 0.047583818435668945], [0.4995841383934021, 0.01382436789572239, 0.0017661130987107754, 0.08258343487977982], [0.42873167991638184, 0.27258726954460144, 0.20817619562149048, 0.4982690215110779], [0.24455808103084564, 0.42871490120887756, 0.45946192741394043, 0.36610716581344604], [0.05611858889460564, 0.20213836431503296, 0.16467683017253876, 0.09433294087648392], [0.07387426495552063, 0.3788360059261322, 0.16099436581134796, 0.3784846365451813]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8cf32724159a136b56691ea6c925bfa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6bbf425d31b170c975112cae854aadd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a87bbe6f9c0968c05f91b7a69992f19f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ecc303f8ddfb9bd09e4b17cc7bee71f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ed45fe8b53310a8bc4ca8c11ca76950c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7af88b4e4a2a446f8915c0dbb62f30bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_33eabdc762f69494d5ced96ce3257e08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85641fc0a9c8a7b32d0053b7be1b007c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.29233288764953613, 0.36788859963417053, 0.006307649426162243, 0.2807435095310211], [0.3606562316417694, 0.492276668548584, 0.20333808660507202, 0.06060683727264404], [0.14438258111476898, 0.47764408588409424, 0.38392844796180725, 0.3385300934314728], [0.037337061017751694, 0.3944132924079895, 0.4904634356498718, 0.26613932847976685], [0.280810683965683, 0.27498728036880493, 0.3712540864944458, 0.056545354425907135], [0.42267999053001404, 0.3459642827510834, 0.30062058568000793, 0.436982125043869]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([6], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9f12d4fa7bf610eb1da7cb2914f8fdfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51d402daab72be882d05e4f57bacef27
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_21b731e6f3997e40ea1b443a13600c2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d0365a47f1f4b9ab59d7f90970ba8fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c864bcdca29d412ac7feb3649df87e01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_543794d4d0417e476e2bac89c6ecbbdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_18cd4dfd9b786aeb481577b3ac31dba2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_99bbd83c81f1e7fa270bd14ad39bb007(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18cd4dfd9b786aeb481577b3ac31dba2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([300], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_35d82f8552c0d29b22c89f9e18f0e0f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c0443fb3fef0b480b9c1fb68b8a83d02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35d82f8552c0d29b22c89f9e18f0e0f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_8ef9c7c5cb0c58793b3abad0c8333194(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e20ccd1f6031eadb695bbf29cd1103c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef9c7c5cb0c58793b3abad0c8333194
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_9c22e1eaa0f1d425de6c1e8a49dbb17c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2c66d463bb85ee894ab259cb08315d30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c22e1eaa0f1d425de6c1e8a49dbb17c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_b70557adf78dc78348e431158f5fb6a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8dd8a7d58c8f7d214fa324af1138fc50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b70557adf78dc78348e431158f5fb6a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_392cec73a40d83556fa50843ebc38b65(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c83b5929b9bce197afe130a2a9867c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392cec73a40d83556fa50843ebc38b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_e74e4ac2a1e86c2762765bba5db3f09c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e61fd2e7a0461a55da7905e2070c2e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e74e4ac2a1e86c2762765bba5db3f09c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_420db7d151f7d03ac57bdf0349713ac4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_749a23fc937882e30b344d4d04fe487b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420db7d151f7d03ac57bdf0349713ac4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d2aeefb05961c65e350a13cba41fa8e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b70557adf78dc78348e431158f5fb6a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.13313806056976318, 0.4626670777797699, 0.46212807297706604, 0.25356119871139526], [0.12604497373104095, 0.47608232498168945, 0.47727617621421814, 0.3902834355831146]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_570ede2bc95baba5ab884b2fbfe78ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392cec73a40d83556fa50843ebc38b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cc5a2941b64ca9c90438e8ed5461a374(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e74e4ac2a1e86c2762765bba5db3f09c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0847e6da424baefb10d2874139af2740(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420db7d151f7d03ac57bdf0349713ac4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_496e94243da9986c6abdd075d050c3e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18cd4dfd9b786aeb481577b3ac31dba2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([100], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f715388c2989041c5daa91c612571aa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35d82f8552c0d29b22c89f9e18f0e0f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e01799e0fc823f9ec95cd03e5b8d94ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef9c7c5cb0c58793b3abad0c8333194
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_25daee7b34f69dd08cd1def2f50bc8c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c22e1eaa0f1d425de6c1e8a49dbb17c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6cc1d5b01d9e8aa71234f5887875cb35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b70557adf78dc78348e431158f5fb6a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.08681345731019974, 0.18166474997997284, 0.365119993686676, 0.0067316764034330845], [0.10514085739850998, 0.4515710771083832, 0.0026658447459340096, 0.3969408869743347]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8e67d4b7cc225a7a1468200ad2c912f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392cec73a40d83556fa50843ebc38b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 68, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c03ebfd2524fdbe3c7b8c86cfa2f537f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e74e4ac2a1e86c2762765bba5db3f09c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 34, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0141ed62dfa447e6c20d90cf267d498a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420db7d151f7d03ac57bdf0349713ac4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7a909013a8c7b508d1dacb6d53f8633e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b70557adf78dc78348e431158f5fb6a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.47624194622039795, 0.4252462089061737, 0.4653246998786926, 0.4718214273452759], [0.26620176434516907, 0.39858391880989075, 0.11420700699090958, 0.23182038962841034]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7765ce186280d9374ca4eddccdf08552(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392cec73a40d83556fa50843ebc38b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5fc24b61e772df02539dd3ea29868155(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e74e4ac2a1e86c2762765bba5db3f09c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_38ed21b032167eed7db26704a6ae94bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420db7d151f7d03ac57bdf0349713ac4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8dd8a7d58c8f7d214fa324af1138fc50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b70557adf78dc78348e431158f5fb6a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9c83b5929b9bce197afe130a2a9867c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392cec73a40d83556fa50843ebc38b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4e61fd2e7a0461a55da7905e2070c2e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e74e4ac2a1e86c2762765bba5db3f09c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_749a23fc937882e30b344d4d04fe487b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420db7d151f7d03ac57bdf0349713ac4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d63ed5b24686c1559be9543b533331af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b70557adf78dc78348e431158f5fb6a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.49739229679107666, 0.3351207375526428, 0.4484364688396454, 0.12505020201206207], [0.030114924535155296, 0.19822652637958527, 0.2196095734834671, 0.40188801288604736], [0.37833085656166077, 0.005445803515613079, 0.4010592997074127, 0.42676693201065063], [0.35176393389701843, 0.16488398611545563, 0.10220362991094589, 0.2679820656776428], [0.20115259289741516, 0.23507601022720337, 0.2623601257801056, 0.21225251257419586], [0.3057858347892761, 0.18095499277114868, 0.42079871892929077, 0.12342901527881622], [0.1707804948091507, 0.18216678500175476, 0.05291115120053291, 0.3348720669746399]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_51c161c8de28f08d23792ac9c8d8d6c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392cec73a40d83556fa50843ebc38b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_90fb93ebbf44fc2a32918a3b4592f28a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e74e4ac2a1e86c2762765bba5db3f09c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_25e8ead7b9167a015b87325b25ebd358(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420db7d151f7d03ac57bdf0349713ac4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_7e5b2bf16c8552a944a31f45a2e0a41c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fe3463031e186cc93ff3bfcc3ccc808d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e5b2bf16c8552a944a31f45a2e0a41c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.12931282818317413, 0.48861533403396606, 0.17718838155269623, 0.12520994246006012], [0.07523564994335175, 0.4386076331138611, 0.3620389699935913, 0.07658398151397705], [0.14756430685520172, 0.07572465389966965, 0.26838070154190063, 0.37414035201072693], [0.3915621042251587, 0.19104118645191193, 0.01597001403570175, 0.05103715509176254], [0.22588086128234863, 0.0050764307379722595, 0.31528225541114807, 0.298330157995224], [0.177719846367836, 0.25171852111816406, 0.10758128017187119, 0.3024943470954895]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([6], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_40e8d29e701505e66bd5a4552c59912a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f7b128689f1ac65856b3b996cae24a6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40e8d29e701505e66bd5a4552c59912a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_c104a01f6b3c1f82e9603098f52e3112(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7bcf504a754413b72e3d7f3bb08b2514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c104a01f6b3c1f82e9603098f52e3112
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_8265afe31c4300027a1dac01f4c83172(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_374a2ab749c8493cd36da64fffa14bfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8265afe31c4300027a1dac01f4c83172
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5db058c95e800c12f6a6d4deac5b3788(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b70557adf78dc78348e431158f5fb6a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.1701001077890396, 0.014843899756669998, 0.09852124750614166, 0.3393995761871338], [0.11748797446489334, 0.2589859068393707, 0.31148210167884827, 0.10683373361825943], [0.3116267919540405, 0.4089503586292267, 0.36851558089256287, 0.13546034693717957]], dtype='float32').reshape([3, 4]),
                paddle.to_tensor([3], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_28bfdc51aacbb7b8649895feea3eaad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392cec73a40d83556fa50843ebc38b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 136], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8c5bdadfce65462891e654194032e5df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e74e4ac2a1e86c2762765bba5db3f09c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 68], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_911a5b95b57706ce908b3b7405eafd81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420db7d151f7d03ac57bdf0349713ac4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0fd1c49d1e415cd508208e8b0057dc6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b70557adf78dc78348e431158f5fb6a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.049569591879844666, 0.30775371193885803, 0.3094034790992737, 0.13927102088928223], [0.24713994562625885, 0.37523457407951355, 0.024224577471613884, 0.3512221872806549]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7765ce186280d9374ca4eddccdf08552(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392cec73a40d83556fa50843ebc38b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5fc24b61e772df02539dd3ea29868155(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e74e4ac2a1e86c2762765bba5db3f09c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_38ed21b032167eed7db26704a6ae94bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420db7d151f7d03ac57bdf0349713ac4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_40d0c0a10f69ccb44aa41be445cb757c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e5b2bf16c8552a944a31f45a2e0a41c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.03567306697368622, 0.41631054878234863, 0.14406552910804749, 0.48773443698883057]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d05683e74eae9df54138bd58c59c6100(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40e8d29e701505e66bd5a4552c59912a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0631ded3eb400ea0e16a7767d81462b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c104a01f6b3c1f82e9603098f52e3112
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_00853a7d7a8dfc197074fdfa171bed91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8265afe31c4300027a1dac01f4c83172
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4d03ded3dd6174d954673ad678f14da9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b70557adf78dc78348e431158f5fb6a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.33109039068222046, 0.08781209588050842, 0.023874972015619278, 0.4068280756473541], [0.27392226457595825, 0.46916234493255615, 0.3789480924606323, 0.32833385467529297], [0.3113239109516144, 0.34468361735343933, 0.4564971327781677, 0.373810350894928], [0.20377878844738007, 0.48164132237434387, 0.3123581111431122, 0.20703266561031342], [0.44531503319740295, 0.4649951756000519, 0.18325623869895935, 0.19394664466381073], [0.322318434715271, 0.3634970784187317, 0.12563112378120422, 0.49102601408958435], [0.2974511981010437, 0.2236485332250595, 0.2726476192474365, 0.3250909149646759]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6f4bf12903db38f149b48ebf2a2d7e7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392cec73a40d83556fa50843ebc38b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 68, 104], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_04ab950d97058af88da8599833131e1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e74e4ac2a1e86c2762765bba5db3f09c
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 34, 52], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_194f0e1a45ecf0bdda6669f9397394ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420db7d151f7d03ac57bdf0349713ac4
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_adf425b3bc4f149fc62c32b7e8234425(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e5b2bf16c8552a944a31f45a2e0a41c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4272349774837494, 0.47877711057662964, 0.08770851790904999, 0.08134938776493073]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f7b128689f1ac65856b3b996cae24a6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40e8d29e701505e66bd5a4552c59912a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7bcf504a754413b72e3d7f3bb08b2514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c104a01f6b3c1f82e9603098f52e3112
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_374a2ab749c8493cd36da64fffa14bfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8265afe31c4300027a1dac01f4c83172
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a5d8882124d58836ec1775de56de15b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b70557adf78dc78348e431158f5fb6a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.2777930796146393, 0.02657657489180565, 0.47445645928382874, 0.20409588515758514], [0.1387108713388443, 0.39539310336112976, 0.1251879781484604, 0.12581704556941986], [0.0013369943480938673, 0.06631842255592346, 0.38734182715415955, 0.07199130952358246], [0.4746476411819458, 0.005258830729871988, 0.14967621862888336, 0.14246855676174164], [0.15963207185268402, 0.3332064747810364, 0.09021896123886108, 0.44095081090927124]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([5], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7386516cf7e86290fd2dc593ab9abc5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392cec73a40d83556fa50843ebc38b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e89e876c284e071a66084d9a52c7223a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e74e4ac2a1e86c2762765bba5db3f09c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7f21f6f9f3b134ab547b1e34a567e366(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420db7d151f7d03ac57bdf0349713ac4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_77ec9cb6c890f2ebe8b909d5d1b0a425(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b70557adf78dc78348e431158f5fb6a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.36710721254348755, 0.2733869254589081, 0.12741735577583313, 0.09983520209789276], [0.1689828783273697, 0.12775754928588867, 0.4929034411907196, 0.024316783994436264], [0.28232342004776, 0.3154520094394684, 0.34447357058525085, 0.1899302452802658], [0.3894905745983124, 0.21266524493694305, 0.15552382171154022, 0.2049335241317749], [0.14210514724254608, 0.42684876918792725, 0.16730007529258728, 0.03494313359260559], [0.36946699023246765, 0.25839653611183167, 0.40660321712493896, 0.049302250146865845], [0.4219869375228882, 0.046291451901197433, 0.45390188694000244, 0.2299237996339798]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_603fb7d751166118f29db801c34724bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392cec73a40d83556fa50843ebc38b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6f0981af39e57eaf1e244aa6498db116(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e74e4ac2a1e86c2762765bba5db3f09c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c6e9335176e991a17e1d361d117c3623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420db7d151f7d03ac57bdf0349713ac4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b7fbae9a2c40b21a834b14e1538d92bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b70557adf78dc78348e431158f5fb6a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.04564209654927254, 0.4040566384792328, 0.4310336410999298, 0.19761954247951508], [0.06333372741937637, 0.015020381659269333, 0.26050060987472534, 0.4925742447376251], [0.004450720734894276, 0.09676632285118103, 0.15310807526111603, 0.2296193391084671], [0.3995705544948578, 0.20686708390712738, 0.16882404685020447, 0.41516441106796265], [0.10278435796499252, 0.2873406708240509, 0.33401980996131897, 0.45226940512657166], [0.16829141974449158, 0.3355092406272888, 0.43085777759552, 0.3128385841846466], [0.39289021492004395, 0.4308372735977173, 0.11628603935241699, 0.23280969262123108]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_51c161c8de28f08d23792ac9c8d8d6c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392cec73a40d83556fa50843ebc38b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_90fb93ebbf44fc2a32918a3b4592f28a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e74e4ac2a1e86c2762765bba5db3f09c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_25e8ead7b9167a015b87325b25ebd358(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420db7d151f7d03ac57bdf0349713ac4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cc09ef99ebc899ced09677131f44cbf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e5b2bf16c8552a944a31f45a2e0a41c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.06026938185095787, 0.10351205617189407, 0.010360955260694027, 0.05767056718468666]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_affcc68fa3e27e0d77d8d1533f546d88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40e8d29e701505e66bd5a4552c59912a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_edfbc64fd860360b2cfaef118bf5c580(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c104a01f6b3c1f82e9603098f52e3112
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7a61e8b21d34c0d6cd0afdf11c9f7d3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8265afe31c4300027a1dac01f4c83172
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_99bbd83c81f1e7fa270bd14ad39bb007(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18cd4dfd9b786aeb481577b3ac31dba2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([300], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c0443fb3fef0b480b9c1fb68b8a83d02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35d82f8552c0d29b22c89f9e18f0e0f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e20ccd1f6031eadb695bbf29cd1103c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef9c7c5cb0c58793b3abad0c8333194
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2c66d463bb85ee894ab259cb08315d30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c22e1eaa0f1d425de6c1e8a49dbb17c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cedd8d22fb811e7a64be8797f82a9ac2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b70557adf78dc78348e431158f5fb6a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.11586524546146393, 0.3802587389945984, 0.2655188739299774, 0.32558372616767883], [0.2183745801448822, 0.062354110181331635, 0.09305060654878616, 0.19626672565937042], [0.051588743925094604, 0.2933014929294586, 0.06933026015758514, 0.34952929615974426], [0.4719468057155609, 0.09559851884841919, 0.3501664698123932, 0.04593765735626221], [0.3326735496520996, 0.0424429252743721, 0.33680230379104614, 0.1775297373533249]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([5], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7386516cf7e86290fd2dc593ab9abc5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392cec73a40d83556fa50843ebc38b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e89e876c284e071a66084d9a52c7223a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e74e4ac2a1e86c2762765bba5db3f09c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7f21f6f9f3b134ab547b1e34a567e366(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420db7d151f7d03ac57bdf0349713ac4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c926ecab3e1351aa8b72c34900095d51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b70557adf78dc78348e431158f5fb6a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3226352334022522, 0.1387707144021988, 0.13630889356136322, 0.3679457902908325]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f032f26a6949ff02d0bba6adeb73f90a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392cec73a40d83556fa50843ebc38b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ad8a878067c214ffb48c192bc10a59aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e74e4ac2a1e86c2762765bba5db3f09c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_057c68a327901fd0691cb88f93e7ebaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420db7d151f7d03ac57bdf0349713ac4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f1ea956a986a1aaca20baa957bc54518(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e5b2bf16c8552a944a31f45a2e0a41c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4692842364311218, 0.07899530231952667, 0.43798011541366577, 0.0650915876030922]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f403099ed168551b644e653d6cdabcaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40e8d29e701505e66bd5a4552c59912a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_94fb4c94c9ea89ae630bc8a5e318c9a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c104a01f6b3c1f82e9603098f52e3112
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_83993f2563c36e0cc10c2a23e339dd42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8265afe31c4300027a1dac01f4c83172
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f0935e0192dade8f1a098454ded6ce3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e5b2bf16c8552a944a31f45a2e0a41c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_affcc68fa3e27e0d77d8d1533f546d88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40e8d29e701505e66bd5a4552c59912a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_edfbc64fd860360b2cfaef118bf5c580(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c104a01f6b3c1f82e9603098f52e3112
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7a61e8b21d34c0d6cd0afdf11c9f7d3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8265afe31c4300027a1dac01f4c83172
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_496e94243da9986c6abdd075d050c3e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18cd4dfd9b786aeb481577b3ac31dba2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([100], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f715388c2989041c5daa91c612571aa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35d82f8552c0d29b22c89f9e18f0e0f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e01799e0fc823f9ec95cd03e5b8d94ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef9c7c5cb0c58793b3abad0c8333194
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_25daee7b34f69dd08cd1def2f50bc8c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c22e1eaa0f1d425de6c1e8a49dbb17c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1614bce3d0d3025867e6d36b0db0f176(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e5b2bf16c8552a944a31f45a2e0a41c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.2400294840335846, 0.33962640166282654, 0.2767137289047241, 0.09159930795431137]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_629dc16cfa89770cf0f8ee37e770ba2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40e8d29e701505e66bd5a4552c59912a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2d4d3c3be7c30e33eb8dbced11d2a067(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c104a01f6b3c1f82e9603098f52e3112
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_70be126ca5f2d9d2ea06ec39d6535fe9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8265afe31c4300027a1dac01f4c83172
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_23f4c96c87abdbcfae8844588070400c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b70557adf78dc78348e431158f5fb6a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.13706611096858978, 0.08502600342035294, 0.44341975450515747, 0.3082224726676941], [0.009993945248425007, 0.3372018337249756, 0.11831139028072357, 0.047583818435668945], [0.4995841383934021, 0.01382436789572239, 0.0017661130987107754, 0.08258343487977982], [0.42873167991638184, 0.27258726954460144, 0.20817619562149048, 0.4982690215110779], [0.24455808103084564, 0.42871490120887756, 0.45946192741394043, 0.36610716581344604], [0.05611858889460564, 0.20213836431503296, 0.16467683017253876, 0.09433294087648392], [0.07387426495552063, 0.3788360059261322, 0.16099436581134796, 0.3784846365451813]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_603fb7d751166118f29db801c34724bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392cec73a40d83556fa50843ebc38b65
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6f0981af39e57eaf1e244aa6498db116(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e74e4ac2a1e86c2762765bba5db3f09c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c6e9335176e991a17e1d361d117c3623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420db7d151f7d03ac57bdf0349713ac4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9df1a0c20e3ba379b14505a0002447bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e5b2bf16c8552a944a31f45a2e0a41c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.29233288764953613, 0.36788859963417053, 0.006307649426162243, 0.2807435095310211], [0.3606562316417694, 0.492276668548584, 0.20333808660507202, 0.06060683727264404], [0.14438258111476898, 0.47764408588409424, 0.38392844796180725, 0.3385300934314728], [0.037337061017751694, 0.3944132924079895, 0.4904634356498718, 0.26613932847976685], [0.280810683965683, 0.27498728036880493, 0.3712540864944458, 0.056545354425907135], [0.42267999053001404, 0.3459642827510834, 0.30062058568000793, 0.436982125043869]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([6], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d05683e74eae9df54138bd58c59c6100(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40e8d29e701505e66bd5a4552c59912a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0631ded3eb400ea0e16a7767d81462b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c104a01f6b3c1f82e9603098f52e3112
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_00853a7d7a8dfc197074fdfa171bed91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8265afe31c4300027a1dac01f4c83172
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()