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
    class PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f700b93744d26168fbf4aea858d99360(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8920a44ef02ee3e80bf75b37fb5ffdfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0f6ee66477202fed089d8150efb1062f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e2106517a1895a122c7cc0b8d277c804(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_52380ce33348f01227c28de64b2b8397(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_770fb7d88ea80304c1fd21030a884f69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9bbdd020f5d205af657a57531239b03b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_78669814f2b9f317a0a99273314e1eb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8550bc8f7d82f49e7bb8866db7d0d5b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c21f15224d95274563080369532c8c93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ebed57eb3dc8a19017eb2e17cade9e86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_200055d32d25bdf604f79b31092ca297(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e07dec664838553256c656611f34733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_200055d32d25bdf604f79b31092ca297
        def get_inputs(self):
            return [
                paddle.to_tensor(1025, dtype='int32').reshape([]),
                paddle.to_tensor(1025, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_949634f170a609f76e22695a9bf81adf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_edf101ffe7f497d5a1f1a7b1321cce86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e18c63e0c974a1c7300737a16aa3789f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_66a6c958509d3fdfa66f344160deab9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e7b2647850ff02a169c068fed3ad7916(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3320c26e636450f1e2aaf281c363042b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2c28eff5ace2edb61d3d967e5b522fb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6c543a55afaf6718500ed7ae64ceee44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_8578a55737e6e3f3506d6d28fe0dfb57(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86970], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e2076f2b7a825e2a0c2c79e1dfd959d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8578a55737e6e3f3506d6d28fe0dfb57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_2c140ff71e0ec2fcf784f33f0a79df5d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[242991], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e4e7702538d52307a5b2bdcb494e283(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c140ff71e0ec2fcf784f33f0a79df5d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_4a16a2c6a388067ff68481df77bc6e10(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[220968], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea275234fe471ff0886b276c96a8f13f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a16a2c6a388067ff68481df77bc6e10
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_f52aba9b8c685082c64a0d8867725c18(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[153450], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a66395c0862dc70def9c60a7dcc4ed66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f52aba9b8c685082c64a0d8867725c18
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_80d62e442aa3a832091c4f43e0fdc0bc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2002], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c36022f110e72707fa36a8696c8cb961(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80d62e442aa3a832091c4f43e0fdc0bc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_92cda1d553a875afbb35be05b3396024(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80d62e442aa3a832091c4f43e0fdc0bc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_0b0426222fbb89a103036c374189f9cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[185691], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fec38b6efab82ef605979429b451b741(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b0426222fbb89a103036c374189f9cc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_d319a5b9a8c32bd9f9e66a463fd6ab62(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1021], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_321f08f6abdc03160c4ebcc5ea8a9ed3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d319a5b9a8c32bd9f9e66a463fd6ab62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0148d6db35ec1d75aa26ed300de19097(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d319a5b9a8c32bd9f9e66a463fd6ab62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_246b87c159e12e47441931231e185c04(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1002], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_25e90f6414a5139fc0d0ecb3c685b7db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_246b87c159e12e47441931231e185c04
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d360629a574ec8695c13646989389686(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_246b87c159e12e47441931231e185c04
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0e07dec664838553256c656611f34733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_200055d32d25bdf604f79b31092ca297
        def get_inputs(self):
            return [
                paddle.to_tensor(1025, dtype='int32').reshape([]),
                paddle.to_tensor(1025, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_da29422922034523fac731acb9d5dcdf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[113061], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_507629f65ececeae29f22ae7bfe95424(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da29422922034523fac731acb9d5dcdf
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_bca0d2801a5551f23bf6bf204251297b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[205923], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_43f0c5a581b06ed11be7c93ebbc516e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bca0d2801a5551f23bf6bf204251297b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_8928ef7cf8fbab3df0ce5aa9fd5a772a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[123783], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_321db28467a076029b211306f9d594ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8928ef7cf8fbab3df0ce5aa9fd5a772a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_80a0862f44f8526bb458b59480c93efc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171888], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8c70395ae683496191d5178c4b67958f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80a0862f44f8526bb458b59480c93efc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_853ed02d4eed6a54118683880dc95bb2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[217413], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb462493123977ddbfcb5fe4e58d8e87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_853ed02d4eed6a54118683880dc95bb2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_145256b055c4dcde31b101a82d059e60(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1027], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7f59c8fe91df789be428a6be818526dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_145256b055c4dcde31b101a82d059e60
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3dc538bf08399b3092223e350d3d7ce6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_145256b055c4dcde31b101a82d059e60
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_252ea10e689ee25917245634a8b92522(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[185658], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cf8fc8c0f2c55e4fc6a843c444a57faf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_252ea10e689ee25917245634a8b92522
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f700b93744d26168fbf4aea858d99360(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8920a44ef02ee3e80bf75b37fb5ffdfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0f6ee66477202fed089d8150efb1062f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e2106517a1895a122c7cc0b8d277c804(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_52380ce33348f01227c28de64b2b8397(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_770fb7d88ea80304c1fd21030a884f69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9bbdd020f5d205af657a57531239b03b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_78669814f2b9f317a0a99273314e1eb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8550bc8f7d82f49e7bb8866db7d0d5b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c21f15224d95274563080369532c8c93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ebed57eb3dc8a19017eb2e17cade9e86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0e07dec664838553256c656611f34733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_200055d32d25bdf604f79b31092ca297
        def get_inputs(self):
            return [
                paddle.to_tensor(1025, dtype='int32').reshape([]),
                paddle.to_tensor(1025, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_949634f170a609f76e22695a9bf81adf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_edf101ffe7f497d5a1f1a7b1321cce86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e18c63e0c974a1c7300737a16aa3789f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_66a6c958509d3fdfa66f344160deab9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e7b2647850ff02a169c068fed3ad7916(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3320c26e636450f1e2aaf281c363042b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2c28eff5ace2edb61d3d967e5b522fb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6c543a55afaf6718500ed7ae64ceee44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abb2fb9ca728da2cc804166fdf4808ef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    

if __name__ == '__main__':
    unittest.main()