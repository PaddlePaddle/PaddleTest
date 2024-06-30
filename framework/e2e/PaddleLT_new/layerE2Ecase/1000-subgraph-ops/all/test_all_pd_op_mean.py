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
    class PrimitiveOp_fef4825bd7aa567e81d6804046c74cb5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f90b866b699161a6561c06f5e9542fb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fef4825bd7aa567e81d6804046c74cb5
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_85818af49d650d7b7de99206ec0a4adc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aac572a686b5f04f84df0e3f683d09b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85818af49d650d7b7de99206ec0a4adc
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 152, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dddcfdba8b17a8c752088b4f2c88028c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_96c8a2c3becb36d4e1d2c4f5d6a48160(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dddcfdba8b17a8c752088b4f2c88028c
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5e986a862427c1921486f47943b9f658(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c421352cc00f00b2c391ea6794b7790(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e986a862427c1921486f47943b9f658
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5a88b619d11c86798d14fffcd915d2d4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_80cf6d5a8414b8a443fcb28c9638512e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a88b619d11c86798d14fffcd915d2d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bf6330779b0c55ab8046fa4738fb5a76(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8fc14754bab6697d42c5d4650f5a0135(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf6330779b0c55ab8046fa4738fb5a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2651bd99e903c49a7fa4976a36c616e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a88b619d11c86798d14fffcd915d2d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_93049e4e047b7ca7c05160cc7b6d6a20(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f26ae7f2452ca7bbff6279080787cee2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93049e4e047b7ca7c05160cc7b6d6a20
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbd60143c47bdb8eaba13e9abc65c056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf6330779b0c55ab8046fa4738fb5a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a435f038cb22a98efa20f0369c080a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf6330779b0c55ab8046fa4738fb5a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee9cfdb63daeef33f4680ca30d89630a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf6330779b0c55ab8046fa4738fb5a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f9e76fbb369fe29587345a5ccea75b13(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [-1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5e2827d0428f2ae502abe867c6ca97f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9e76fbb369fe29587345a5ccea75b13
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07dac2c98d99cd5dc333de0663e6f9e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a88b619d11c86798d14fffcd915d2d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69457f9d2ab245b4016927f5a8dbd018(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93049e4e047b7ca7c05160cc7b6d6a20
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c3b7acb3c40f19bdf7dc652f8b31aeb0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41db8c102f6dc13bbe1ba46cfe3e713c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3b7acb3c40f19bdf7dc652f8b31aeb0
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b97e20a481bb7d73130bba84b284f93b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dddcfdba8b17a8c752088b4f2c88028c
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53f2789a6bf37b00b69e23f36a84fd6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9e76fbb369fe29587345a5ccea75b13
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fe739e29cfb07f4be4bbc097a48a50c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3b7acb3c40f19bdf7dc652f8b31aeb0
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8fcb1e492c0e5731b46a1996aa20e377(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a88b619d11c86798d14fffcd915d2d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2d0cdc0b3ebb1ab83e840195ffe9c62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85818af49d650d7b7de99206ec0a4adc
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88cae99d94d0e0d972bb93a51fae0671(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93049e4e047b7ca7c05160cc7b6d6a20
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1f5f30958e425b9f3bbccf04671a77fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d299bdfd11a0f5dcc42c3cd5fc88e04b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f5f30958e425b9f3bbccf04671a77fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 168, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c59f3466e388d4c5d640f114deab4046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf6330779b0c55ab8046fa4738fb5a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8ccca5ca9fcf16529592b471d399894b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b161ad2b62f02df3eae07a87f8c70e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b161ad2b62f02df3eae07a87f8c70e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba73658e6b74d7a81fa064899ba05076(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba73658e6b74d7a81fa064899ba05076(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77e4b6e52ac8d1b89a9bf2fc8026b6bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77e4b6e52ac8d1b89a9bf2fc8026b6bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3156de2575ecbeaa290aaa624ae57d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3156de2575ecbeaa290aaa624ae57d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c3aca2b012f2047e77e4ee1faea4c238(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [], False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_978ade9d7ef97b3d8bf19af947508aa3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3aca2b012f2047e77e4ee1faea4c238
        def get_inputs(self):
            return [
                paddle.to_tensor([1.056071162223816, 1.1451103687286377, 4.043917655944824, 1.4260247945785522, 1.2243378162384033, 1.826979160308838], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_1710410d19130f642b8fa2f6ef9e5386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1710410d19130f642b8fa2f6ef9e5386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2397d00685d44daf93b94b2c95f32472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2397d00685d44daf93b94b2c95f32472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b930e475b8ed4019ab11abc745ce4282(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b930e475b8ed4019ab11abc745ce4282(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc407f52567460c6e72da9bdb246c87d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc407f52567460c6e72da9bdb246c87d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3556ad8b810b65e7811f488d231f31a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6d393c396c12aaff05b7612769124d1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3556ad8b810b65e7811f488d231f31a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79073b852187675a6a14aff33d09a229(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9e76fbb369fe29587345a5ccea75b13
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ce4ab35f1cfd60d15eb5036117b06ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93049e4e047b7ca7c05160cc7b6d6a20
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cfd821594d2e6062df49de24f611cad8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a88b619d11c86798d14fffcd915d2d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68064472830ad5d5941f884117de4dd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dddcfdba8b17a8c752088b4f2c88028c
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1710410d19130f642b8fa2f6ef9e5386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1710410d19130f642b8fa2f6ef9e5386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2397d00685d44daf93b94b2c95f32472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2397d00685d44daf93b94b2c95f32472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b930e475b8ed4019ab11abc745ce4282(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b930e475b8ed4019ab11abc745ce4282(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc407f52567460c6e72da9bdb246c87d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc407f52567460c6e72da9bdb246c87d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73c27523cb212b3cc2b1e881d88e0fc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9e76fbb369fe29587345a5ccea75b13
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0e46a39db95567bf6d172eabdba3b006(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7cdc078bed33e29403165fc62ea4ecc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e46a39db95567bf6d172eabdba3b006
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70fe75ab4adb0bc10497db58bfd2de41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85818af49d650d7b7de99206ec0a4adc
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 136, 136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0e26a89a6425275f8b2e45b99921378(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a88b619d11c86798d14fffcd915d2d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_61de9cf4c35c3fd43c47fca90c383ae5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 4, 13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_51df9f38c7fa6e27920ea4354b933351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61de9cf4c35c3fd43c47fca90c383ae5
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d78a0586d92f29fd64bb24a26dd4d39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93049e4e047b7ca7c05160cc7b6d6a20
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a408bd10489f43f35bd26ae4417ea5f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85818af49d650d7b7de99206ec0a4adc
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4c3bd549be613be029c384dd6cff026(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93049e4e047b7ca7c05160cc7b6d6a20
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 184, 184], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6944091a1ef888ce36ad1e442321168(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf6330779b0c55ab8046fa4738fb5a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a829cc871a8789ac62110ef8adfe57a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_398c6339710a44b7983ace2f6b1dbab5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a829cc871a8789ac62110ef8adfe57a4
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30966dad6a43fd930833c5ba994c6ca8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93049e4e047b7ca7c05160cc7b6d6a20
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6bf5a7198db7bf2b4b355b54d82bf29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9e76fbb369fe29587345a5ccea75b13
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f141c86a0183920512ac586c6ec1d243(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fef4825bd7aa567e81d6804046c74cb5
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_999083b055baf78afa674ade2a9531b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3556ad8b810b65e7811f488d231f31a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c3caacca85eab493ce8ed6564290c2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9e76fbb369fe29587345a5ccea75b13
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05c7f619dc777748c7541551d5d3460c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a88b619d11c86798d14fffcd915d2d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7e54dad741f1fa3c8d73611e8bfeb5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9e76fbb369fe29587345a5ccea75b13
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dacc0eb59edb3cafd51c04bd6f43010(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a88b619d11c86798d14fffcd915d2d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_36b6a53777a2266f2abdd46d39e023dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_02560cc134a6fd8720703821d7fbfd26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36b6a53777a2266f2abdd46d39e023dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3aa5b43d7ba95948b9e845cea42b1f6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a88b619d11c86798d14fffcd915d2d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f219144bad89007a7db6aaa2113ea43a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dddcfdba8b17a8c752088b4f2c88028c
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f210c079d53280612095fef5451928a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e46a39db95567bf6d172eabdba3b006
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad9257b43402df296caf381247fc468b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85818af49d650d7b7de99206ec0a4adc
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b161ad2b62f02df3eae07a87f8c70e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b161ad2b62f02df3eae07a87f8c70e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba73658e6b74d7a81fa064899ba05076(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba73658e6b74d7a81fa064899ba05076(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77e4b6e52ac8d1b89a9bf2fc8026b6bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77e4b6e52ac8d1b89a9bf2fc8026b6bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3156de2575ecbeaa290aaa624ae57d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3156de2575ecbeaa290aaa624ae57d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c42ca6952e9c3a1f22906392d1f2b65d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dddcfdba8b17a8c752088b4f2c88028c
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf7a118c0046bef2879c61f5e9dbd156(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9e76fbb369fe29587345a5ccea75b13
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba8659a671443b0cf0f6dab3d5e92a53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf6330779b0c55ab8046fa4738fb5a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97f1dae41c3619bdac2d2f687390c7c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9e76fbb369fe29587345a5ccea75b13
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58b086b07956689e6059ce98ad9b7f53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9e76fbb369fe29587345a5ccea75b13
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_768246a2b6428df49fb1508429f283b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf6330779b0c55ab8046fa4738fb5a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7e3b2b075998f50f322b3ecc0a2c5ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf6330779b0c55ab8046fa4738fb5a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4760c0a916238510daf51e23349f1a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93049e4e047b7ca7c05160cc7b6d6a20
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef4b66265386e04b11265a642df8d3af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93049e4e047b7ca7c05160cc7b6d6a20
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_adea7b98025be65c2f2571eb42e57638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a88b619d11c86798d14fffcd915d2d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d2295b7628b35fc4d7de2561bd993075(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 4, 50, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4a2ad6f95a1ea1818caeedda6a110e59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2295b7628b35fc4d7de2561bd993075
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21c902531ac251be3e15116139a37e6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a88b619d11c86798d14fffcd915d2d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68b10f7f313de16b143ac53d8ec5cf5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e46a39db95567bf6d172eabdba3b006
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_691295a0ee8236cba9a48649093b2749(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36b6a53777a2266f2abdd46d39e023dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec19f9e6034cdc56e459fedab8f0f9a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85818af49d650d7b7de99206ec0a4adc
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41bc97d6a9ffef40640c19526bb17e8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3556ad8b810b65e7811f488d231f31a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc1c16c412f735a29c02b8435015072d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3b7acb3c40f19bdf7dc652f8b31aeb0
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3c03386c981dc5cbed313f3a6632ed8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf6330779b0c55ab8046fa4738fb5a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4976cdf3daef64dde59527f73b7760b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9e76fbb369fe29587345a5ccea75b13
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28a0985adc1ba1228714f49c4b4d3d99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e986a862427c1921486f47943b9f658
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78d8caa6098b98d4a55fbee3c5ea6b44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a829cc871a8789ac62110ef8adfe57a4
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_929e4b7c852a34b63b34716cbaeee107(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf6330779b0c55ab8046fa4738fb5a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24f04d23c57a6d73d3ca1a60cbd3dd07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f5f30958e425b9f3bbccf04671a77fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c5e0cf05bfd2343f6eac28ac794f19f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93049e4e047b7ca7c05160cc7b6d6a20
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5785488f6809c18c075863fcdeb972ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 4, 25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5d7cf43820d70d7cea6cd8ab9b48b54b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5785488f6809c18c075863fcdeb972ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b32c66641889a93c42c259c6afe6ca03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93049e4e047b7ca7c05160cc7b6d6a20
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aa48938fac6663b02850ac5d857ec22f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 4, 7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4cc5a13b1898afc84a2522ccd3c1a0a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa48938fac6663b02850ac5d857ec22f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac2e618cd7a4b4730d7c2844d448a77c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9e76fbb369fe29587345a5ccea75b13
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0792279cedf39ced33a63a89da078ab0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 4, 100, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_83fe6879b3c52decc243a06d5ac098e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0792279cedf39ced33a63a89da078ab0
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80b83d096d09a87905801f91f110316e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e986a862427c1921486f47943b9f658
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_696839563a86b74925f30ce0789f1197(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0b1213e63e3133cd4ff48f8d137a750(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a00b75adc52d6a56b239d040fbda3a68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 152, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e059537b5ed9b1d1614833f4206c8a5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e145249789d9663f41b1288c08555894(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b52c8548d1f0f28de84ce08bd4e0b52a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf299ae45624664cb0e8a926e318aa14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24bd1215dc6d0f534293028eacb2f987(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93711642ae9b35128d513dc055de4783(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7146e5271ac0446f91468fb507c70868(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_695fc918937055edb23441c5523b2a9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e0fccde0e31d963f51459bdd9457b9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3e25490cbb5c6b07851a6dc792f6f85a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [-1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_416d81c58aab2c05b517baa21e7ff177(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e25490cbb5c6b07851a6dc792f6f85a
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96e27398197aff7d68ff56951efc1b24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3757fa096a8fd03c82198c25429e4182(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c80a9a0e485586377c8c19da481d2bb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_091c2d67893b3c8086b2f09cbb037700(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcabc9fd35c1043e22c19556544f5ec8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e25490cbb5c6b07851a6dc792f6f85a
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4566dffd090fd64564f35d114c3c9a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e3b69d878621a6e4ef3e5ab58c15a79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_edf2468a3a2039186f9e213c2c6c97cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6af2ac7e3a74bb407284d270482090d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0d1e10b2dca531742ce12ded3fed53f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 168, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a96dac51982b955ff81c5cc43ef12af2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b161ad2b62f02df3eae07a87f8c70e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b161ad2b62f02df3eae07a87f8c70e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba73658e6b74d7a81fa064899ba05076(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba73658e6b74d7a81fa064899ba05076(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77e4b6e52ac8d1b89a9bf2fc8026b6bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77e4b6e52ac8d1b89a9bf2fc8026b6bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3156de2575ecbeaa290aaa624ae57d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3156de2575ecbeaa290aaa624ae57d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_978ade9d7ef97b3d8bf19af947508aa3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3aca2b012f2047e77e4ee1faea4c238
        def get_inputs(self):
            return [
                paddle.to_tensor([1.056071162223816, 1.1451103687286377, 4.043917655944824, 1.4260247945785522, 1.2243378162384033, 1.826979160308838], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_1710410d19130f642b8fa2f6ef9e5386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1710410d19130f642b8fa2f6ef9e5386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2397d00685d44daf93b94b2c95f32472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2397d00685d44daf93b94b2c95f32472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b930e475b8ed4019ab11abc745ce4282(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b930e475b8ed4019ab11abc745ce4282(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc407f52567460c6e72da9bdb246c87d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc407f52567460c6e72da9bdb246c87d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1fa7fab5cb36ab262db0399891f8311(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfb0a1e75933fb37fdddbeb7e5148185(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e25490cbb5c6b07851a6dc792f6f85a
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71da9d15dec732c65d7f9540f767ea09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c2f8ecfc8337247271d3e4101daaed1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd15d4aa6b0be15ce4292ea863bba131(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1710410d19130f642b8fa2f6ef9e5386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1710410d19130f642b8fa2f6ef9e5386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2397d00685d44daf93b94b2c95f32472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2397d00685d44daf93b94b2c95f32472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b930e475b8ed4019ab11abc745ce4282(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b930e475b8ed4019ab11abc745ce4282(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc407f52567460c6e72da9bdb246c87d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc407f52567460c6e72da9bdb246c87d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c5ab40a87e26053d068de67e54e91d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e25490cbb5c6b07851a6dc792f6f85a
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef7f6ec51572d93f2c96d33e4d9a862e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6795d6042421cf37dcdcc97a82856955(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 136, 136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c39bd99e38e71f227068df4ad0f4b0b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8c48c2fbf92e8b3d95c280089bbad187(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.mean(input_0, [2], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e2220343f1aaff3a978ab8316d7b0989(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c48c2fbf92e8b3d95c280089bbad187
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f180d9affa9dcda44214aed7c465d10c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6d0435595b17d369b2ff8c59af00233(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea505fe76508c1f520c409c69b43a1db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 184, 184], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_289f10df3dd56f3ad9278d0485065779(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8aa2d76dfd58ea3a6c6667f37c10456a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e378b84fbc7a85b16a1d6b4c7eb4a08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1424be8d6984da5683d041a11b4a0b24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e25490cbb5c6b07851a6dc792f6f85a
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cab3a1945365ebcdfd91919f1a2bbdec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e180a36b6a704996f1b93a5f2488695(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be7e495dcc1ab288d53b1f48d8ae7c09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e25490cbb5c6b07851a6dc792f6f85a
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_269830f99bcdf27d46daf0de43bd6e05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4fc465f95a65e9c3ce8c5ef40f10add(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e25490cbb5c6b07851a6dc792f6f85a
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdecc886c17a07903c3eef6c575195d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05f6590fff662001d4ac525ae33cf3ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_960e0eb8f71b92624a23befff39b09e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87f0c1f8009023309c0cffb11279aa0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cdcc2c9eced69c5372f656c97eb72be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6409d542f9485a8203dbb6999cfe372(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b161ad2b62f02df3eae07a87f8c70e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b161ad2b62f02df3eae07a87f8c70e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba73658e6b74d7a81fa064899ba05076(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba73658e6b74d7a81fa064899ba05076(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77e4b6e52ac8d1b89a9bf2fc8026b6bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77e4b6e52ac8d1b89a9bf2fc8026b6bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3156de2575ecbeaa290aaa624ae57d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3156de2575ecbeaa290aaa624ae57d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ccca5ca9fcf16529592b471d399894b
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff5ea32f7ab80d620c27417439e9f3ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c13c409b2a205e3607c24e592704eaa5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e25490cbb5c6b07851a6dc792f6f85a
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efc2bf498826bc8946845fdc122b7ae6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbd5b419d8546f8ba1e199a1cef52362(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e25490cbb5c6b07851a6dc792f6f85a
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9f6c3daa7e6a00656699a30023e27d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e25490cbb5c6b07851a6dc792f6f85a
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce7ce09f0901a40c181f3345dc740ccf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83a392f16de0a4015b8c73622a6871b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5094db58608a03d516f1767604d42f38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e902aa3cbb52004242932164b7614e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45a5303ff9e400eabaa8d6f82859cfb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2d237af43ae6aed3b00806cf079afd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c48c2fbf92e8b3d95c280089bbad187
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c0aefb70244b064820d420e740d02a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61cb4f43e986fb3e97d39d59a0acd495(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3dcbb883a9ff99ac4f77077fffbde43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4da704b986ffd3464c043562be21388b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dfeb58868db6c49a68b56280afcb40b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf3081e9939236f6142e7295a9a80f23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6016a3388ca31231973ccbf24e6a7eab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aae08fdc8c8a23f5c65bf7f16b99e8b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e25490cbb5c6b07851a6dc792f6f85a
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b06bbacb70e5d697c5cf0c2fc14e1da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd1c76ce688a8d9aa5fe7dbc4341ee57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f26289e4911af2faddbff0562f511cea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_057dc4d63dfb31e9ae715200c752027d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb893376b15e8e36fdd528ee193c9e79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0735116d06bcb5be4063143751203c4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c48c2fbf92e8b3d95c280089bbad187
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65d4420f6bd6f76b1981f277f7ff4bed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2517fda32c71f74cbc30eeb24289d60d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c48c2fbf92e8b3d95c280089bbad187
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9a710537b71f1f14a156eb18969e3a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e25490cbb5c6b07851a6dc792f6f85a
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea8cc16108df87ba0ce9420ef8d837f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c48c2fbf92e8b3d95c280089bbad187
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_457a3c2612c392cdc833e2286bc391d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696839563a86b74925f30ce0789f1197
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()