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
    class PrimitiveOp_90758931b249bd84b1ebf5fba3c8df2b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2c4420531bc38c772c232eacf13a5c91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90758931b249bd84b1ebf5fba3c8df2b
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e577b886f79bd903c251a357ebdc5773(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 576, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eafff98ebc5d31ed37d23beeda57376e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e577b886f79bd903c251a357ebdc5773
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_477fcf0a2a334f35de5792243018d12d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e4fb631234c26fe544907c446fe6f57e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_477fcf0a2a334f35de5792243018d12d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_daa8bb846808c1f34292beee0899ca80(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b72c46720e3520babb74f31a5d99515b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa8bb846808c1f34292beee0899ca80
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e069f04c76216b0530835e4f3d727ba2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_70e4f675fe4fbc072dbd33238e5adcec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e069f04c76216b0530835e4f3d727ba2
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0a8e4d08cce199df0bf5ea7d38a16781(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_510b60bdbe1b16426464e3a0950976da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a8e4d08cce199df0bf5ea7d38a16781
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_383b54930a96139b5f2d5594e78a718d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_501c4e816ec7068075b9ffb5077357a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_383b54930a96139b5f2d5594e78a718d
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fa380d8ff6f18fb2ec906a24171f2102(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_875c7356900d806901c383328c0a3e27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa380d8ff6f18fb2ec906a24171f2102
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_065838b45ebaa2d88ff3bf59f05675c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a1ab591cf09943099cba4d8aaa233c98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_065838b45ebaa2d88ff3bf59f05675c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a887cd667b759f2e63559f00fe237038(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e2670a2891b31171d03b7f60c6248bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a887cd667b759f2e63559f00fe237038
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_529724f180a3e385bf148cc5f75fda73(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1bd01bbe866fc59df8ec7c692fea39f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_529724f180a3e385bf148cc5f75fda73
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e3bf66337728cbb33e5884e7ffd73111(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_943d8d06d6b741cdae9ec3e177c7d4f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3bf66337728cbb33e5884e7ffd73111
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fc46373009926a6e9b5671d87b09f6dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_67dada53b5f36a35badbf9133f40d5a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc46373009926a6e9b5671d87b09f6dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bd01bbe866fc59df8ec7c692fea39f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_529724f180a3e385bf148cc5f75fda73
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2c354b8435c4dd1eacd966155589233c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6f041aa81c64813bd4873af8ad8d54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c354b8435c4dd1eacd966155589233c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2670a2891b31171d03b7f60c6248bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a887cd667b759f2e63559f00fe237038
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_144baa1c749885c7c35a8db8af7e114f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c7346dbbab60cc376ad15e5a900c8d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_144baa1c749885c7c35a8db8af7e114f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7346dbbab60cc376ad15e5a900c8d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_144baa1c749885c7c35a8db8af7e114f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6f041aa81c64813bd4873af8ad8d54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c354b8435c4dd1eacd966155589233c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6f041aa81c64813bd4873af8ad8d54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c354b8435c4dd1eacd966155589233c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_683b7274983cb70b74303c80e1af8e5b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_996d0a73de6e4dba68f8f919393610e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_683b7274983cb70b74303c80e1af8e5b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.1285748481750488]], [[1.2519538402557373]], [[1.3091199398040771]], [[0.8894222974777222]], [[1.5531479120254517]], [[0.9988418817520142]], [[1.1143465042114258]], [[1.5623719692230225]], [[1.5880999565124512]], [[1.3541173934936523]], [[1.0930718183517456]], [[1.2446036338806152]], [[1.55824613571167]], [[0.9519383907318115]], [[1.7797831296920776]], [[1.385715126991272]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_b6f041aa81c64813bd4873af8ad8d54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c354b8435c4dd1eacd966155589233c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7fdd3a987f3719e64d0072fe692dd585(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 44, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1ea502b6b12490395b93509016f55b05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fdd3a987f3719e64d0072fe692dd585
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b72c46720e3520babb74f31a5d99515b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa8bb846808c1f34292beee0899ca80
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bd01bbe866fc59df8ec7c692fea39f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_529724f180a3e385bf148cc5f75fda73
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c4420531bc38c772c232eacf13a5c91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90758931b249bd84b1ebf5fba3c8df2b
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_501c4e816ec7068075b9ffb5077357a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_383b54930a96139b5f2d5594e78a718d
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_510b60bdbe1b16426464e3a0950976da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a8e4d08cce199df0bf5ea7d38a16781
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_75d43e255a3dc013baa30a97d91d538c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_46b559529589ed3e082108dc565c2474(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75d43e255a3dc013baa30a97d91d538c
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_510b60bdbe1b16426464e3a0950976da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a8e4d08cce199df0bf5ea7d38a16781
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ed0f19a5df077d5a5a82fb675200e07d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_923120a1476cd6bc4c97495b4d5c201a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed0f19a5df077d5a5a82fb675200e07d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c4420531bc38c772c232eacf13a5c91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90758931b249bd84b1ebf5fba3c8df2b
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1ab591cf09943099cba4d8aaa233c98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_065838b45ebaa2d88ff3bf59f05675c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f4526bf495a8207bbc1193a8c054f17(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 320, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c29acbe5760c037626f4be557c4976b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f4526bf495a8207bbc1193a8c054f17
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a30b6d08605bf29059e8649d50405ba7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_495c84ba0851c73866aaecbe9d3617f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a30b6d08605bf29059e8649d50405ba7
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bd01bbe866fc59df8ec7c692fea39f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_529724f180a3e385bf148cc5f75fda73
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_510b60bdbe1b16426464e3a0950976da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a8e4d08cce199df0bf5ea7d38a16781
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_83c7338f566ce75b780ea55ac1a5caf1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4943e0486efff578cb4381d6e71ae066(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83c7338f566ce75b780ea55ac1a5caf1
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2670a2891b31171d03b7f60c6248bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a887cd667b759f2e63559f00fe237038
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b47ca28fb4eabea5c58286ed75ee29c5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e5319f8c3ffaee030c0fdabe4f84ab43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b47ca28fb4eabea5c58286ed75ee29c5
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bc809da70bde370f7cbc8a6338db9dfb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cbd9ccffcb1bd7c6ba2e4f4864628a3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc809da70bde370f7cbc8a6338db9dfb
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_510b60bdbe1b16426464e3a0950976da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a8e4d08cce199df0bf5ea7d38a16781
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c29acbe5760c037626f4be557c4976b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f4526bf495a8207bbc1193a8c054f17
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7346dbbab60cc376ad15e5a900c8d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_144baa1c749885c7c35a8db8af7e114f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cb0aa91b0cbbfbd4240711b77f2f946d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce52a48689887a6af0dd79df9736aa32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb0aa91b0cbbfbd4240711b77f2f946d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_510b60bdbe1b16426464e3a0950976da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a8e4d08cce199df0bf5ea7d38a16781
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_875c7356900d806901c383328c0a3e27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa380d8ff6f18fb2ec906a24171f2102
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b47b0c215a6b8896072ab5b8e55e8f8c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 576, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e6fce93b7e790d65f80c8cc0f10f5a22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b47b0c215a6b8896072ab5b8e55e8f8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e4f675fe4fbc072dbd33238e5adcec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e069f04c76216b0530835e4f3d727ba2
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eafff98ebc5d31ed37d23beeda57376e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e577b886f79bd903c251a357ebdc5773
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6fce93b7e790d65f80c8cc0f10f5a22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b47b0c215a6b8896072ab5b8e55e8f8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2670a2891b31171d03b7f60c6248bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a887cd667b759f2e63559f00fe237038
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_510b60bdbe1b16426464e3a0950976da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a8e4d08cce199df0bf5ea7d38a16781
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5319f8c3ffaee030c0fdabe4f84ab43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b47ca28fb4eabea5c58286ed75ee29c5
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67dada53b5f36a35badbf9133f40d5a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc46373009926a6e9b5671d87b09f6dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7346dbbab60cc376ad15e5a900c8d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_144baa1c749885c7c35a8db8af7e114f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_875c7356900d806901c383328c0a3e27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa380d8ff6f18fb2ec906a24171f2102
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4fb631234c26fe544907c446fe6f57e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_477fcf0a2a334f35de5792243018d12d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7346dbbab60cc376ad15e5a900c8d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_144baa1c749885c7c35a8db8af7e114f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6f041aa81c64813bd4873af8ad8d54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c354b8435c4dd1eacd966155589233c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67dada53b5f36a35badbf9133f40d5a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc46373009926a6e9b5671d87b09f6dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1ab591cf09943099cba4d8aaa233c98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_065838b45ebaa2d88ff3bf59f05675c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6f041aa81c64813bd4873af8ad8d54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c354b8435c4dd1eacd966155589233c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6f041aa81c64813bd4873af8ad8d54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c354b8435c4dd1eacd966155589233c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_923120a1476cd6bc4c97495b4d5c201a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed0f19a5df077d5a5a82fb675200e07d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5ed79a9e59ed399dbcf88257f1de97f7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_43cc7b6649024c68bdc5fa627bc0e64a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ed79a9e59ed399dbcf88257f1de97f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7346dbbab60cc376ad15e5a900c8d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_144baa1c749885c7c35a8db8af7e114f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2670a2891b31171d03b7f60c6248bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a887cd667b759f2e63559f00fe237038
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e4f675fe4fbc072dbd33238e5adcec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e069f04c76216b0530835e4f3d727ba2
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_510b60bdbe1b16426464e3a0950976da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a8e4d08cce199df0bf5ea7d38a16781
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c113c13d99dd85962042d241a087465d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 288, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4667a11b25dc5f0c3fbad027fa1ce33e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c113c13d99dd85962042d241a087465d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4fb631234c26fe544907c446fe6f57e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_477fcf0a2a334f35de5792243018d12d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2670a2891b31171d03b7f60c6248bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a887cd667b759f2e63559f00fe237038
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67dada53b5f36a35badbf9133f40d5a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc46373009926a6e9b5671d87b09f6dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c7a8f66a2b987df1062213a6aa5e0257(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_db5b7fe7827b5b1ae745423a88ae61c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7a8f66a2b987df1062213a6aa5e0257
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b72c46720e3520babb74f31a5d99515b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa8bb846808c1f34292beee0899ca80
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7346dbbab60cc376ad15e5a900c8d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_144baa1c749885c7c35a8db8af7e114f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_707648da7cd249ff6089b7ace3e79075(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_35f7c50e5b9b2353f73eca8a86bc2d06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_707648da7cd249ff6089b7ace3e79075
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce52a48689887a6af0dd79df9736aa32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb0aa91b0cbbfbd4240711b77f2f946d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67dada53b5f36a35badbf9133f40d5a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc46373009926a6e9b5671d87b09f6dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4fb631234c26fe544907c446fe6f57e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_477fcf0a2a334f35de5792243018d12d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7346dbbab60cc376ad15e5a900c8d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_144baa1c749885c7c35a8db8af7e114f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a999445cd592a6f868202949ac156e50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_683b7274983cb70b74303c80e1af8e5b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.0142029523849487]], [[2.053401470184326]], [[1.3052400350570679]], [[1.1951097249984741]], [[2.430665969848633]], [[1.4276739358901978]], [[1.5479648113250732]], [[1.6883649826049805]], [[2.279353618621826]], [[1.985144853591919]], [[1.8742430210113525]], [[1.6576496362686157]], [[1.9620004892349243]], [[1.9623486995697021]], [[1.26690673828125]], [[1.4913731813430786]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_b6f041aa81c64813bd4873af8ad8d54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c354b8435c4dd1eacd966155589233c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db5b7fe7827b5b1ae745423a88ae61c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7a8f66a2b987df1062213a6aa5e0257
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4943e0486efff578cb4381d6e71ae066(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83c7338f566ce75b780ea55ac1a5caf1
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c69e2832c4de38f7cf8c3016dbd282b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 400, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b8bf3de862f09c50c52e86b7b6d228b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69e2832c4de38f7cf8c3016dbd282b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5319f8c3ffaee030c0fdabe4f84ab43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b47ca28fb4eabea5c58286ed75ee29c5
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ca53e43804298c2a08814deac1f5e4bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0561b79099b573dd0b543cedaae61efb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca53e43804298c2a08814deac1f5e4bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c4420531bc38c772c232eacf13a5c91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90758931b249bd84b1ebf5fba3c8df2b
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6f041aa81c64813bd4873af8ad8d54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c354b8435c4dd1eacd966155589233c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_510b60bdbe1b16426464e3a0950976da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a8e4d08cce199df0bf5ea7d38a16781
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbd9ccffcb1bd7c6ba2e4f4864628a3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc809da70bde370f7cbc8a6338db9dfb
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4e860cb18cb41ed42dd2d96698cb81b9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a658e5bfe11e62bac84cb07699249c1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e860cb18cb41ed42dd2d96698cb81b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7346dbbab60cc376ad15e5a900c8d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_144baa1c749885c7c35a8db8af7e114f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4fb631234c26fe544907c446fe6f57e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_477fcf0a2a334f35de5792243018d12d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbd9ccffcb1bd7c6ba2e4f4864628a3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc809da70bde370f7cbc8a6338db9dfb
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ea502b6b12490395b93509016f55b05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fdd3a987f3719e64d0072fe692dd585
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce52a48689887a6af0dd79df9736aa32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb0aa91b0cbbfbd4240711b77f2f946d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b8bf3de862f09c50c52e86b7b6d228b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69e2832c4de38f7cf8c3016dbd282b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6c4594d444346c5f0132ff86210a4403(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 56, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1855ca5a381d3e881d77e01261e58458(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c4594d444346c5f0132ff86210a4403
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2670a2891b31171d03b7f60c6248bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a887cd667b759f2e63559f00fe237038
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c4420531bc38c772c232eacf13a5c91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90758931b249bd84b1ebf5fba3c8df2b
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43cc7b6649024c68bdc5fa627bc0e64a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ed79a9e59ed399dbcf88257f1de97f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b72c46720e3520babb74f31a5d99515b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa8bb846808c1f34292beee0899ca80
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbd9ccffcb1bd7c6ba2e4f4864628a3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc809da70bde370f7cbc8a6338db9dfb
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2670a2891b31171d03b7f60c6248bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a887cd667b759f2e63559f00fe237038
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8b85ea8d90ad133c2f9448c393ebf48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3bf66337728cbb33e5884e7ffd73111
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8b85ea8d90ad133c2f9448c393ebf48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3bf66337728cbb33e5884e7ffd73111
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8b85ea8d90ad133c2f9448c393ebf48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3bf66337728cbb33e5884e7ffd73111
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8b85ea8d90ad133c2f9448c393ebf48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3bf66337728cbb33e5884e7ffd73111
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fbaa736a7167cc30122d4a91e729df77(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2752d2d3ccf94355700506abd4e9a509(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbaa736a7167cc30122d4a91e729df77
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2752d2d3ccf94355700506abd4e9a509(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbaa736a7167cc30122d4a91e729df77
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2752d2d3ccf94355700506abd4e9a509(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbaa736a7167cc30122d4a91e729df77
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2752d2d3ccf94355700506abd4e9a509(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbaa736a7167cc30122d4a91e729df77
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce52a48689887a6af0dd79df9736aa32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb0aa91b0cbbfbd4240711b77f2f946d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6f041aa81c64813bd4873af8ad8d54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c354b8435c4dd1eacd966155589233c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_501c4e816ec7068075b9ffb5077357a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_383b54930a96139b5f2d5594e78a718d
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_501c4e816ec7068075b9ffb5077357a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_383b54930a96139b5f2d5594e78a718d
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_501c4e816ec7068075b9ffb5077357a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_383b54930a96139b5f2d5594e78a718d
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_802a1bcf27ca1c717f0fdd773c21966c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 200, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d61d414db394c2cf9fd45ee7fb3f7d48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_802a1bcf27ca1c717f0fdd773c21966c
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db5b7fe7827b5b1ae745423a88ae61c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7a8f66a2b987df1062213a6aa5e0257
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2670a2891b31171d03b7f60c6248bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a887cd667b759f2e63559f00fe237038
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_501c4e816ec7068075b9ffb5077357a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_383b54930a96139b5f2d5594e78a718d
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4fb631234c26fe544907c446fe6f57e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_477fcf0a2a334f35de5792243018d12d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4fb631234c26fe544907c446fe6f57e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_477fcf0a2a334f35de5792243018d12d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_495c84ba0851c73866aaecbe9d3617f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a30b6d08605bf29059e8649d50405ba7
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5319f8c3ffaee030c0fdabe4f84ab43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b47ca28fb4eabea5c58286ed75ee29c5
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_79ed09e3b30c614e3bef5e3f27c3728c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 288, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6daac0c2a7c458ff2825858c7c8e96d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79ed09e3b30c614e3bef5e3f27c3728c
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2670a2891b31171d03b7f60c6248bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a887cd667b759f2e63559f00fe237038
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e4f675fe4fbc072dbd33238e5adcec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e069f04c76216b0530835e4f3d727ba2
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4667a11b25dc5f0c3fbad027fa1ce33e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c113c13d99dd85962042d241a087465d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4fb631234c26fe544907c446fe6f57e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_477fcf0a2a334f35de5792243018d12d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_875c7356900d806901c383328c0a3e27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa380d8ff6f18fb2ec906a24171f2102
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e4f675fe4fbc072dbd33238e5adcec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e069f04c76216b0530835e4f3d727ba2
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6f041aa81c64813bd4873af8ad8d54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c354b8435c4dd1eacd966155589233c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6f041aa81c64813bd4873af8ad8d54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c354b8435c4dd1eacd966155589233c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6f041aa81c64813bd4873af8ad8d54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c354b8435c4dd1eacd966155589233c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d61d414db394c2cf9fd45ee7fb3f7d48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_802a1bcf27ca1c717f0fdd773c21966c
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7346dbbab60cc376ad15e5a900c8d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_144baa1c749885c7c35a8db8af7e114f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c29acbe5760c037626f4be557c4976b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f4526bf495a8207bbc1193a8c054f17
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7346dbbab60cc376ad15e5a900c8d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_144baa1c749885c7c35a8db8af7e114f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce52a48689887a6af0dd79df9736aa32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb0aa91b0cbbfbd4240711b77f2f946d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2670a2891b31171d03b7f60c6248bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a887cd667b759f2e63559f00fe237038
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7346dbbab60cc376ad15e5a900c8d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_144baa1c749885c7c35a8db8af7e114f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f781e5a187d7fc4704910692badfd93d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d22b731bbe2c955a9e4792057898c4ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f781e5a187d7fc4704910692badfd93d
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2670a2891b31171d03b7f60c6248bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a887cd667b759f2e63559f00fe237038
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce52a48689887a6af0dd79df9736aa32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb0aa91b0cbbfbd4240711b77f2f946d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67dada53b5f36a35badbf9133f40d5a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc46373009926a6e9b5671d87b09f6dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4667a11b25dc5f0c3fbad027fa1ce33e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c113c13d99dd85962042d241a087465d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_510b60bdbe1b16426464e3a0950976da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a8e4d08cce199df0bf5ea7d38a16781
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_510b60bdbe1b16426464e3a0950976da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a8e4d08cce199df0bf5ea7d38a16781
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5319f8c3ffaee030c0fdabe4f84ab43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b47ca28fb4eabea5c58286ed75ee29c5
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_501c4e816ec7068075b9ffb5077357a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_383b54930a96139b5f2d5594e78a718d
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b72c46720e3520babb74f31a5d99515b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa8bb846808c1f34292beee0899ca80
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4fb631234c26fe544907c446fe6f57e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_477fcf0a2a334f35de5792243018d12d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46b559529589ed3e082108dc565c2474(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75d43e255a3dc013baa30a97d91d538c
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_923120a1476cd6bc4c97495b4d5c201a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed0f19a5df077d5a5a82fb675200e07d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_58a50741464023f0cb7fd727261b01ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b5da08a86751ce66bf39bdf692f11ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58a50741464023f0cb7fd727261b01ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a658e5bfe11e62bac84cb07699249c1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e860cb18cb41ed42dd2d96698cb81b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43cc7b6649024c68bdc5fa627bc0e64a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ed79a9e59ed399dbcf88257f1de97f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2670a2891b31171d03b7f60c6248bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a887cd667b759f2e63559f00fe237038
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1855ca5a381d3e881d77e01261e58458(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c4594d444346c5f0132ff86210a4403
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6fce93b7e790d65f80c8cc0f10f5a22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b47b0c215a6b8896072ab5b8e55e8f8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbd9ccffcb1bd7c6ba2e4f4864628a3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc809da70bde370f7cbc8a6338db9dfb
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c29acbe5760c037626f4be557c4976b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f4526bf495a8207bbc1193a8c054f17
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_501c4e816ec7068075b9ffb5077357a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_383b54930a96139b5f2d5594e78a718d
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bd01bbe866fc59df8ec7c692fea39f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_529724f180a3e385bf148cc5f75fda73
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6f041aa81c64813bd4873af8ad8d54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c354b8435c4dd1eacd966155589233c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b72c46720e3520babb74f31a5d99515b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa8bb846808c1f34292beee0899ca80
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b5da08a86751ce66bf39bdf692f11ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58a50741464023f0cb7fd727261b01ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_875c7356900d806901c383328c0a3e27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa380d8ff6f18fb2ec906a24171f2102
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0561b79099b573dd0b543cedaae61efb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca53e43804298c2a08814deac1f5e4bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6f041aa81c64813bd4873af8ad8d54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c354b8435c4dd1eacd966155589233c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1ab591cf09943099cba4d8aaa233c98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_065838b45ebaa2d88ff3bf59f05675c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d61d414db394c2cf9fd45ee7fb3f7d48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_802a1bcf27ca1c717f0fdd773c21966c
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b8bf3de862f09c50c52e86b7b6d228b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c69e2832c4de38f7cf8c3016dbd282b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7346dbbab60cc376ad15e5a900c8d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_144baa1c749885c7c35a8db8af7e114f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7346dbbab60cc376ad15e5a900c8d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_144baa1c749885c7c35a8db8af7e114f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_495c84ba0851c73866aaecbe9d3617f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a30b6d08605bf29059e8649d50405ba7
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4fb631234c26fe544907c446fe6f57e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_477fcf0a2a334f35de5792243018d12d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce52a48689887a6af0dd79df9736aa32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb0aa91b0cbbfbd4240711b77f2f946d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b72c46720e3520babb74f31a5d99515b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa8bb846808c1f34292beee0899ca80
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c29acbe5760c037626f4be557c4976b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f4526bf495a8207bbc1193a8c054f17
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_923120a1476cd6bc4c97495b4d5c201a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed0f19a5df077d5a5a82fb675200e07d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b72c46720e3520babb74f31a5d99515b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa8bb846808c1f34292beee0899ca80
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce52a48689887a6af0dd79df9736aa32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb0aa91b0cbbfbd4240711b77f2f946d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7346dbbab60cc376ad15e5a900c8d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_144baa1c749885c7c35a8db8af7e114f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_943d8d06d6b741cdae9ec3e177c7d4f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3bf66337728cbb33e5884e7ffd73111
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_943d8d06d6b741cdae9ec3e177c7d4f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3bf66337728cbb33e5884e7ffd73111
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_943d8d06d6b741cdae9ec3e177c7d4f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3bf66337728cbb33e5884e7ffd73111
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_943d8d06d6b741cdae9ec3e177c7d4f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3bf66337728cbb33e5884e7ffd73111
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b30ff75df8e1ecf0d6d0b02e9c45f9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbaa736a7167cc30122d4a91e729df77
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[44111.7109375]], [[17871.0390625]], [[34456.4765625]], [[44836.67578125]], [[48451.37109375]], [[51628.41796875]], [[65512.44921875]], [[63779.671875]], [[57183.37890625]], [[47827.90234375]], [[47769.14453125]], [[50280.84765625]], [[47759.7578125]], [[60607.34375]], [[44385.828125]], [[49698.08984375]], [[52658.55078125]], [[57011.4609375]], [[16017.5703125]], [[56370.58984375]], [[56943.859375]], [[51527.203125]], [[46666.2734375]], [[48208.17578125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_8c9ea1a8c4156a6d79554458d443fa2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbaa736a7167cc30122d4a91e729df77
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[27788.734375]], [[65264.19921875]], [[60400.171875]], [[49778.5078125]], [[65997.859375]], [[52837.23828125]], [[76932.9453125]], [[41565.7890625]], [[35542.8125]], [[53729.21484375]], [[76907.5703125]], [[65419.49609375]], [[47146.91015625]], [[61715.30078125]], [[50721.5625]], [[73368.6953125]], [[67275.5]], [[31351.744140625]], [[57100.6953125]], [[44081.046875]], [[52843.109375]], [[40803.4921875]], [[42584.96875]], [[77103.3359375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_dd9a6fa613d8cc60f42392511feefa5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbaa736a7167cc30122d4a91e729df77
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[55397.109375]], [[56351.0078125]], [[43602.48828125]], [[90683.5078125]], [[51676.99609375]], [[58209.51171875]], [[62796.50390625]], [[49609.9140625]], [[62965.30078125]], [[62313.9609375]], [[44644.30078125]], [[51440.71875]], [[47037.28125]], [[53474.1015625]], [[60746.96484375]], [[86316.2734375]], [[50643.44921875]], [[65779.4609375]], [[76064.4296875]], [[57561.7578125]], [[44415.04296875]], [[58072.90625]], [[66083.8046875]], [[58197.02734375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_4d185f22ff815c9fe21b7bf5f4438c2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbaa736a7167cc30122d4a91e729df77
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[52196.23828125]], [[84873.515625]], [[88521.8046875]], [[77473.4140625]], [[56846.40234375]], [[63900.99609375]], [[72542.34375]], [[73128.7890625]], [[36696.59765625]], [[86107.4296875]], [[83204.6796875]], [[41656.7421875]], [[69973.84375]], [[71203.53125]], [[75795.671875]], [[64176.74609375]], [[78790.0625]], [[77350.0703125]], [[75150.515625]], [[71956.296875]], [[78419.7265625]], [[68978.5859375]], [[53145.578125]], [[55659.58203125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_1bd01bbe866fc59df8ec7c692fea39f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_529724f180a3e385bf148cc5f75fda73
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6daac0c2a7c458ff2825858c7c8e96d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79ed09e3b30c614e3bef5e3f27c3728c
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_943d8d06d6b741cdae9ec3e177c7d4f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3bf66337728cbb33e5884e7ffd73111
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_875c7356900d806901c383328c0a3e27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa380d8ff6f18fb2ec906a24171f2102
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_510b60bdbe1b16426464e3a0950976da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a8e4d08cce199df0bf5ea7d38a16781
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c5d42309be11d1a23033fe259012bce5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_399798eafb4a27315e639572bf2f8e55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5d42309be11d1a23033fe259012bce5
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_74114ac8c623756c9f65f1fbdc48e32f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 576, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d4e94981191870617768e0422b2996e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74114ac8c623756c9f65f1fbdc48e32f
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_735a44e2b8e107dcd7ea2e3bcb964d47(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_45f02cc6a381fbd518378ce929dffdc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_735a44e2b8e107dcd7ea2e3bcb964d47
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e8339387f5f7e7d794fc7b2da7c829cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89da2ad51bc789ea8b9e6833f795b69f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8339387f5f7e7d794fc7b2da7c829cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_47c9bf3ce8df9f26d8204b60d8879fd3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e24374c77cb93ca6c957cd7e7f2acd32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47c9bf3ce8df9f26d8204b60d8879fd3
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_733c57f1d7d4a060451326b764e29673(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_806016d67805245d5153b9dbe1c9549a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_733c57f1d7d4a060451326b764e29673
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_40e3c0577f4378c523bf270ee6309ca4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8260446a8bad1849b9e5c55d30d491d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40e3c0577f4378c523bf270ee6309ca4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c0803ba4b5b2be33236c8913b71d383e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4f79ba1cf9740caf15c6e56d9db13b43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0803ba4b5b2be33236c8913b71d383e
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d32ac25801ca64022eec01685d81dd5c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dbd9a85502a5c06a9be41b4a31b80f65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d32ac25801ca64022eec01685d81dd5c
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d8180ff31fd60126804b86435577d630(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f03dd98a3adad661be8a62d60167ee53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8180ff31fd60126804b86435577d630
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9759323007ed977717c997c48e97c505(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eef5083bbc55e13bb4dbc1b6ccf085e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9759323007ed977717c997c48e97c505
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c28fedf580585f5aad19741c7e33b073(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3778c9b2c69d04a5e888e49321d0cdb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c28fedf580585f5aad19741c7e33b073
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b6ef2869f117a1b88c1d1a349088738a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_522d93cb7b3b52cec371f452a758c553(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6ef2869f117a1b88c1d1a349088738a
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eef5083bbc55e13bb4dbc1b6ccf085e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9759323007ed977717c997c48e97c505
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_46d9a3cdc702c2498e3130def8ca9a1d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_43b7f9e56af45d51fe7a729f3976e7fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d9a3cdc702c2498e3130def8ca9a1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f03dd98a3adad661be8a62d60167ee53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8180ff31fd60126804b86435577d630
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_19c6bb9660b9284b2ebf587032bc1342(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_01861fa58e4916ceeee760a2c9bb7f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19c6bb9660b9284b2ebf587032bc1342
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01861fa58e4916ceeee760a2c9bb7f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19c6bb9660b9284b2ebf587032bc1342
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43b7f9e56af45d51fe7a729f3976e7fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d9a3cdc702c2498e3130def8ca9a1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43b7f9e56af45d51fe7a729f3976e7fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d9a3cdc702c2498e3130def8ca9a1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ede35b445820bae8c54181ce6f0325f6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_84cec742a3e0691cc0ebbf36ce480a55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ede35b445820bae8c54181ce6f0325f6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.1285748481750488]], [[1.2519538402557373]], [[1.3091199398040771]], [[0.8894222974777222]], [[1.5531479120254517]], [[0.9988418817520142]], [[1.1143465042114258]], [[1.5623719692230225]], [[1.5880999565124512]], [[1.3541173934936523]], [[1.0930718183517456]], [[1.2446036338806152]], [[1.55824613571167]], [[0.9519383907318115]], [[1.7797831296920776]], [[1.385715126991272]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_43b7f9e56af45d51fe7a729f3976e7fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d9a3cdc702c2498e3130def8ca9a1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a56a18084c9203fcecff70499f204c9c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 44, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7c0863741438117d7de60419dd170a52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a56a18084c9203fcecff70499f204c9c
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89da2ad51bc789ea8b9e6833f795b69f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8339387f5f7e7d794fc7b2da7c829cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eef5083bbc55e13bb4dbc1b6ccf085e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9759323007ed977717c997c48e97c505
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_399798eafb4a27315e639572bf2f8e55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5d42309be11d1a23033fe259012bce5
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8260446a8bad1849b9e5c55d30d491d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40e3c0577f4378c523bf270ee6309ca4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_806016d67805245d5153b9dbe1c9549a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_733c57f1d7d4a060451326b764e29673
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9458d5242bb6b4ee2228b83ad6e7a69a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c6ac5d5925b858705ce1f0a5f29e7ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9458d5242bb6b4ee2228b83ad6e7a69a
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_806016d67805245d5153b9dbe1c9549a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_733c57f1d7d4a060451326b764e29673
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_52575447555da27ecb578f79bd14d455(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_beadfc1071d1111e81f9b98d8171f906(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52575447555da27ecb578f79bd14d455
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_399798eafb4a27315e639572bf2f8e55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5d42309be11d1a23033fe259012bce5
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dbd9a85502a5c06a9be41b4a31b80f65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d32ac25801ca64022eec01685d81dd5c
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c38cd7244f371492b3ea910e34e5dbf3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b9cecb7e9d6bcdc74450285fad3c1ee9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c38cd7244f371492b3ea910e34e5dbf3
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1d566d4220034463669c75be670a7e66(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 100, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aac49494317729409e2b9dd0b27ed397(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d566d4220034463669c75be670a7e66
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eef5083bbc55e13bb4dbc1b6ccf085e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9759323007ed977717c997c48e97c505
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_806016d67805245d5153b9dbe1c9549a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_733c57f1d7d4a060451326b764e29673
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dce03afb1ff4daf6bf4154701db2d9fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c98cad8aea7ab201971893677557808(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dce03afb1ff4daf6bf4154701db2d9fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f03dd98a3adad661be8a62d60167ee53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8180ff31fd60126804b86435577d630
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_29fb0b1a2d50fc3cfc058ebb334fa998(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41daef3a9dbb90e42633e0f0b999de12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29fb0b1a2d50fc3cfc058ebb334fa998
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_175234894bb262b8eeb0c0522de0cf93(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6eb40490cd1e96461d7130648aa0121d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_175234894bb262b8eeb0c0522de0cf93
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_806016d67805245d5153b9dbe1c9549a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_733c57f1d7d4a060451326b764e29673
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9cecb7e9d6bcdc74450285fad3c1ee9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c38cd7244f371492b3ea910e34e5dbf3
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01861fa58e4916ceeee760a2c9bb7f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19c6bb9660b9284b2ebf587032bc1342
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d32a003bab110edcab6a378e807120ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae4e06b869b687b6b7d12375d23d8b54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d32a003bab110edcab6a378e807120ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_806016d67805245d5153b9dbe1c9549a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_733c57f1d7d4a060451326b764e29673
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f79ba1cf9740caf15c6e56d9db13b43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0803ba4b5b2be33236c8913b71d383e
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a5d29028ab6b4d5c5692a384aa0a181(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 576, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9dd282fb7394ca9d0a25dac58d353ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a5d29028ab6b4d5c5692a384aa0a181
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e24374c77cb93ca6c957cd7e7f2acd32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47c9bf3ce8df9f26d8204b60d8879fd3
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4e94981191870617768e0422b2996e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74114ac8c623756c9f65f1fbdc48e32f
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dd282fb7394ca9d0a25dac58d353ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a5d29028ab6b4d5c5692a384aa0a181
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f03dd98a3adad661be8a62d60167ee53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8180ff31fd60126804b86435577d630
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_806016d67805245d5153b9dbe1c9549a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_733c57f1d7d4a060451326b764e29673
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41daef3a9dbb90e42633e0f0b999de12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29fb0b1a2d50fc3cfc058ebb334fa998
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_522d93cb7b3b52cec371f452a758c553(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6ef2869f117a1b88c1d1a349088738a
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01861fa58e4916ceeee760a2c9bb7f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19c6bb9660b9284b2ebf587032bc1342
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f79ba1cf9740caf15c6e56d9db13b43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0803ba4b5b2be33236c8913b71d383e
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45f02cc6a381fbd518378ce929dffdc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_735a44e2b8e107dcd7ea2e3bcb964d47
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01861fa58e4916ceeee760a2c9bb7f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19c6bb9660b9284b2ebf587032bc1342
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43b7f9e56af45d51fe7a729f3976e7fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d9a3cdc702c2498e3130def8ca9a1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_522d93cb7b3b52cec371f452a758c553(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6ef2869f117a1b88c1d1a349088738a
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dbd9a85502a5c06a9be41b4a31b80f65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d32ac25801ca64022eec01685d81dd5c
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43b7f9e56af45d51fe7a729f3976e7fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d9a3cdc702c2498e3130def8ca9a1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43b7f9e56af45d51fe7a729f3976e7fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d9a3cdc702c2498e3130def8ca9a1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beadfc1071d1111e81f9b98d8171f906(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52575447555da27ecb578f79bd14d455
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e0f2b160a752ceca8e4e91e7d7c1cd3c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6da96dc5f570dc93a427f50b7525e233(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0f2b160a752ceca8e4e91e7d7c1cd3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01861fa58e4916ceeee760a2c9bb7f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19c6bb9660b9284b2ebf587032bc1342
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f03dd98a3adad661be8a62d60167ee53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8180ff31fd60126804b86435577d630
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e24374c77cb93ca6c957cd7e7f2acd32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47c9bf3ce8df9f26d8204b60d8879fd3
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_806016d67805245d5153b9dbe1c9549a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_733c57f1d7d4a060451326b764e29673
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0224ca0686a652a445981c37924d14cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bdedb0a9fbf695984bc340f398d9248c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0224ca0686a652a445981c37924d14cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45f02cc6a381fbd518378ce929dffdc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_735a44e2b8e107dcd7ea2e3bcb964d47
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f03dd98a3adad661be8a62d60167ee53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8180ff31fd60126804b86435577d630
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_522d93cb7b3b52cec371f452a758c553(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6ef2869f117a1b88c1d1a349088738a
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4650e92e33b992858700b3462abd1ff1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b17edcaa9daddc43ad52c8b0a487a5d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4650e92e33b992858700b3462abd1ff1
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89da2ad51bc789ea8b9e6833f795b69f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8339387f5f7e7d794fc7b2da7c829cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01861fa58e4916ceeee760a2c9bb7f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19c6bb9660b9284b2ebf587032bc1342
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8b26cb9984940159df99c38cbb9b1b6a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff9579b677d4247558867b9346e70013(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b26cb9984940159df99c38cbb9b1b6a
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae4e06b869b687b6b7d12375d23d8b54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d32a003bab110edcab6a378e807120ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_522d93cb7b3b52cec371f452a758c553(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6ef2869f117a1b88c1d1a349088738a
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45f02cc6a381fbd518378ce929dffdc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_735a44e2b8e107dcd7ea2e3bcb964d47
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01861fa58e4916ceeee760a2c9bb7f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19c6bb9660b9284b2ebf587032bc1342
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9af8096530ce76217ff33300a4e5f75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ede35b445820bae8c54181ce6f0325f6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.0142029523849487]], [[2.053401470184326]], [[1.3052400350570679]], [[1.1951097249984741]], [[2.430665969848633]], [[1.4276739358901978]], [[1.5479648113250732]], [[1.6883649826049805]], [[2.279353618621826]], [[1.985144853591919]], [[1.8742430210113525]], [[1.6576496362686157]], [[1.9620004892349243]], [[1.9623486995697021]], [[1.26690673828125]], [[1.4913731813430786]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_43b7f9e56af45d51fe7a729f3976e7fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d9a3cdc702c2498e3130def8ca9a1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b17edcaa9daddc43ad52c8b0a487a5d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4650e92e33b992858700b3462abd1ff1
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c98cad8aea7ab201971893677557808(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dce03afb1ff4daf6bf4154701db2d9fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5e1802f3960079d3f76f7aa6c0e5711b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 400, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9b5c3e9091912ac7b0305461c639e674(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e1802f3960079d3f76f7aa6c0e5711b
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41daef3a9dbb90e42633e0f0b999de12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29fb0b1a2d50fc3cfc058ebb334fa998
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8002548c2ddc661f9c279c68553ba5a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c53aac094e856bfe5ee6da7ba1e3483(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8002548c2ddc661f9c279c68553ba5a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_399798eafb4a27315e639572bf2f8e55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5d42309be11d1a23033fe259012bce5
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43b7f9e56af45d51fe7a729f3976e7fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d9a3cdc702c2498e3130def8ca9a1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_806016d67805245d5153b9dbe1c9549a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_733c57f1d7d4a060451326b764e29673
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6eb40490cd1e96461d7130648aa0121d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_175234894bb262b8eeb0c0522de0cf93
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_95c11f27437e4b269b9e3886367883d2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 336, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b7e73bd32ed1125e618ef2fc8c62f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95c11f27437e4b269b9e3886367883d2
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01861fa58e4916ceeee760a2c9bb7f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19c6bb9660b9284b2ebf587032bc1342
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45f02cc6a381fbd518378ce929dffdc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_735a44e2b8e107dcd7ea2e3bcb964d47
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6eb40490cd1e96461d7130648aa0121d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_175234894bb262b8eeb0c0522de0cf93
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c0863741438117d7de60419dd170a52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a56a18084c9203fcecff70499f204c9c
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae4e06b869b687b6b7d12375d23d8b54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d32a003bab110edcab6a378e807120ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b5c3e9091912ac7b0305461c639e674(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e1802f3960079d3f76f7aa6c0e5711b
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_11b6549af3838fc11fba259be60ef504(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 56, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_49f6a5ff1161b35fbb82fe874011af14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11b6549af3838fc11fba259be60ef504
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f03dd98a3adad661be8a62d60167ee53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8180ff31fd60126804b86435577d630
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_399798eafb4a27315e639572bf2f8e55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5d42309be11d1a23033fe259012bce5
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6da96dc5f570dc93a427f50b7525e233(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0f2b160a752ceca8e4e91e7d7c1cd3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89da2ad51bc789ea8b9e6833f795b69f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8339387f5f7e7d794fc7b2da7c829cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6eb40490cd1e96461d7130648aa0121d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_175234894bb262b8eeb0c0522de0cf93
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f03dd98a3adad661be8a62d60167ee53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8180ff31fd60126804b86435577d630
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b61a8435e262c6354d5c988dd730987e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1478106ddb222e108cce4e1d2b91b4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b61a8435e262c6354d5c988dd730987e
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1478106ddb222e108cce4e1d2b91b4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b61a8435e262c6354d5c988dd730987e
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1478106ddb222e108cce4e1d2b91b4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b61a8435e262c6354d5c988dd730987e
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1478106ddb222e108cce4e1d2b91b4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b61a8435e262c6354d5c988dd730987e
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a02be5e83d69e921bfd3f635458df2a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 24, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f2dbd001409ad22808716e34d909c380(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a02be5e83d69e921bfd3f635458df2a3
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2dbd001409ad22808716e34d909c380(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a02be5e83d69e921bfd3f635458df2a3
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2dbd001409ad22808716e34d909c380(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a02be5e83d69e921bfd3f635458df2a3
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2dbd001409ad22808716e34d909c380(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a02be5e83d69e921bfd3f635458df2a3
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae4e06b869b687b6b7d12375d23d8b54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d32a003bab110edcab6a378e807120ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43b7f9e56af45d51fe7a729f3976e7fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d9a3cdc702c2498e3130def8ca9a1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8260446a8bad1849b9e5c55d30d491d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40e3c0577f4378c523bf270ee6309ca4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8260446a8bad1849b9e5c55d30d491d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40e3c0577f4378c523bf270ee6309ca4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8260446a8bad1849b9e5c55d30d491d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40e3c0577f4378c523bf270ee6309ca4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aaab3695c766dbaf7ac5f5d673a50f37(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 200, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_49ffcaefdd1863fad30c84842a5ec268(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aaab3695c766dbaf7ac5f5d673a50f37
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b17edcaa9daddc43ad52c8b0a487a5d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4650e92e33b992858700b3462abd1ff1
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f03dd98a3adad661be8a62d60167ee53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8180ff31fd60126804b86435577d630
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8260446a8bad1849b9e5c55d30d491d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40e3c0577f4378c523bf270ee6309ca4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45f02cc6a381fbd518378ce929dffdc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_735a44e2b8e107dcd7ea2e3bcb964d47
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45f02cc6a381fbd518378ce929dffdc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_735a44e2b8e107dcd7ea2e3bcb964d47
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aac49494317729409e2b9dd0b27ed397(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d566d4220034463669c75be670a7e66
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41daef3a9dbb90e42633e0f0b999de12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29fb0b1a2d50fc3cfc058ebb334fa998
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2ab14b2d379857232d90f6495e7412d2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e2d598501cb055fb01ba480b2dbd00e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ab14b2d379857232d90f6495e7412d2
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f03dd98a3adad661be8a62d60167ee53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8180ff31fd60126804b86435577d630
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e24374c77cb93ca6c957cd7e7f2acd32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47c9bf3ce8df9f26d8204b60d8879fd3
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdedb0a9fbf695984bc340f398d9248c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0224ca0686a652a445981c37924d14cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45f02cc6a381fbd518378ce929dffdc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_735a44e2b8e107dcd7ea2e3bcb964d47
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f79ba1cf9740caf15c6e56d9db13b43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0803ba4b5b2be33236c8913b71d383e
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e24374c77cb93ca6c957cd7e7f2acd32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47c9bf3ce8df9f26d8204b60d8879fd3
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43b7f9e56af45d51fe7a729f3976e7fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d9a3cdc702c2498e3130def8ca9a1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43b7f9e56af45d51fe7a729f3976e7fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d9a3cdc702c2498e3130def8ca9a1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43b7f9e56af45d51fe7a729f3976e7fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d9a3cdc702c2498e3130def8ca9a1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49ffcaefdd1863fad30c84842a5ec268(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aaab3695c766dbaf7ac5f5d673a50f37
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01861fa58e4916ceeee760a2c9bb7f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19c6bb9660b9284b2ebf587032bc1342
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9cecb7e9d6bcdc74450285fad3c1ee9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c38cd7244f371492b3ea910e34e5dbf3
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01861fa58e4916ceeee760a2c9bb7f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19c6bb9660b9284b2ebf587032bc1342
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae4e06b869b687b6b7d12375d23d8b54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d32a003bab110edcab6a378e807120ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f03dd98a3adad661be8a62d60167ee53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8180ff31fd60126804b86435577d630
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01861fa58e4916ceeee760a2c9bb7f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19c6bb9660b9284b2ebf587032bc1342
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2d7c4045e09d26bf1285adc16264eaf3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f8baf8e56e3c8171639fcca76c993af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d7c4045e09d26bf1285adc16264eaf3
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f03dd98a3adad661be8a62d60167ee53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8180ff31fd60126804b86435577d630
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae4e06b869b687b6b7d12375d23d8b54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d32a003bab110edcab6a378e807120ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_522d93cb7b3b52cec371f452a758c553(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6ef2869f117a1b88c1d1a349088738a
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdedb0a9fbf695984bc340f398d9248c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0224ca0686a652a445981c37924d14cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_806016d67805245d5153b9dbe1c9549a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_733c57f1d7d4a060451326b764e29673
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_806016d67805245d5153b9dbe1c9549a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_733c57f1d7d4a060451326b764e29673
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41daef3a9dbb90e42633e0f0b999de12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29fb0b1a2d50fc3cfc058ebb334fa998
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8260446a8bad1849b9e5c55d30d491d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40e3c0577f4378c523bf270ee6309ca4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89da2ad51bc789ea8b9e6833f795b69f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8339387f5f7e7d794fc7b2da7c829cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45f02cc6a381fbd518378ce929dffdc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_735a44e2b8e107dcd7ea2e3bcb964d47
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c6ac5d5925b858705ce1f0a5f29e7ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9458d5242bb6b4ee2228b83ad6e7a69a
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beadfc1071d1111e81f9b98d8171f906(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52575447555da27ecb578f79bd14d455
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e36de2fe0d719e5901655d2d5fd72b40(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3bc34dcd38d7bb077018ce67e3213c8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e36de2fe0d719e5901655d2d5fd72b40
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b7e73bd32ed1125e618ef2fc8c62f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95c11f27437e4b269b9e3886367883d2
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6da96dc5f570dc93a427f50b7525e233(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0f2b160a752ceca8e4e91e7d7c1cd3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f03dd98a3adad661be8a62d60167ee53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8180ff31fd60126804b86435577d630
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49f6a5ff1161b35fbb82fe874011af14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11b6549af3838fc11fba259be60ef504
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dd282fb7394ca9d0a25dac58d353ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a5d29028ab6b4d5c5692a384aa0a181
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6eb40490cd1e96461d7130648aa0121d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_175234894bb262b8eeb0c0522de0cf93
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9cecb7e9d6bcdc74450285fad3c1ee9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c38cd7244f371492b3ea910e34e5dbf3
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8260446a8bad1849b9e5c55d30d491d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40e3c0577f4378c523bf270ee6309ca4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eef5083bbc55e13bb4dbc1b6ccf085e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9759323007ed977717c997c48e97c505
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43b7f9e56af45d51fe7a729f3976e7fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d9a3cdc702c2498e3130def8ca9a1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89da2ad51bc789ea8b9e6833f795b69f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8339387f5f7e7d794fc7b2da7c829cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bc34dcd38d7bb077018ce67e3213c8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e36de2fe0d719e5901655d2d5fd72b40
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f79ba1cf9740caf15c6e56d9db13b43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0803ba4b5b2be33236c8913b71d383e
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c53aac094e856bfe5ee6da7ba1e3483(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8002548c2ddc661f9c279c68553ba5a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43b7f9e56af45d51fe7a729f3976e7fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d9a3cdc702c2498e3130def8ca9a1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dbd9a85502a5c06a9be41b4a31b80f65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d32ac25801ca64022eec01685d81dd5c
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49ffcaefdd1863fad30c84842a5ec268(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aaab3695c766dbaf7ac5f5d673a50f37
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b5c3e9091912ac7b0305461c639e674(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e1802f3960079d3f76f7aa6c0e5711b
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01861fa58e4916ceeee760a2c9bb7f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19c6bb9660b9284b2ebf587032bc1342
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01861fa58e4916ceeee760a2c9bb7f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19c6bb9660b9284b2ebf587032bc1342
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aac49494317729409e2b9dd0b27ed397(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d566d4220034463669c75be670a7e66
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45f02cc6a381fbd518378ce929dffdc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_735a44e2b8e107dcd7ea2e3bcb964d47
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae4e06b869b687b6b7d12375d23d8b54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d32a003bab110edcab6a378e807120ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89da2ad51bc789ea8b9e6833f795b69f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8339387f5f7e7d794fc7b2da7c829cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9cecb7e9d6bcdc74450285fad3c1ee9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c38cd7244f371492b3ea910e34e5dbf3
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beadfc1071d1111e81f9b98d8171f906(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52575447555da27ecb578f79bd14d455
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89da2ad51bc789ea8b9e6833f795b69f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8339387f5f7e7d794fc7b2da7c829cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae4e06b869b687b6b7d12375d23d8b54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d32a003bab110edcab6a378e807120ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01861fa58e4916ceeee760a2c9bb7f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19c6bb9660b9284b2ebf587032bc1342
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3778c9b2c69d04a5e888e49321d0cdb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c28fedf580585f5aad19741c7e33b073
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3778c9b2c69d04a5e888e49321d0cdb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c28fedf580585f5aad19741c7e33b073
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3778c9b2c69d04a5e888e49321d0cdb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c28fedf580585f5aad19741c7e33b073
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3778c9b2c69d04a5e888e49321d0cdb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c28fedf580585f5aad19741c7e33b073
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8483df49190666ec3f3815f0e624c17f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 24, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be3f9704d3775dc65a8d4be65823edba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8483df49190666ec3f3815f0e624c17f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[44111.7109375]], [[17871.0390625]], [[34456.4765625]], [[44836.67578125]], [[48451.37109375]], [[51628.41796875]], [[65512.44921875]], [[63779.671875]], [[57183.37890625]], [[47827.90234375]], [[47769.14453125]], [[50280.84765625]], [[47759.7578125]], [[60607.34375]], [[44385.828125]], [[49698.08984375]], [[52658.55078125]], [[57011.4609375]], [[16017.5703125]], [[56370.58984375]], [[56943.859375]], [[51527.203125]], [[46666.2734375]], [[48208.17578125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_4d2174181d8d3719c7368e1c1c70cdcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8483df49190666ec3f3815f0e624c17f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[27788.734375]], [[65264.19921875]], [[60400.171875]], [[49778.5078125]], [[65997.859375]], [[52837.23828125]], [[76932.9453125]], [[41565.7890625]], [[35542.8125]], [[53729.21484375]], [[76907.5703125]], [[65419.49609375]], [[47146.91015625]], [[61715.30078125]], [[50721.5625]], [[73368.6953125]], [[67275.5]], [[31351.744140625]], [[57100.6953125]], [[44081.046875]], [[52843.109375]], [[40803.4921875]], [[42584.96875]], [[77103.3359375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_2ba56cbff2181e5600de08810286d96a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8483df49190666ec3f3815f0e624c17f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[55397.109375]], [[56351.0078125]], [[43602.48828125]], [[90683.5078125]], [[51676.99609375]], [[58209.51171875]], [[62796.50390625]], [[49609.9140625]], [[62965.30078125]], [[62313.9609375]], [[44644.30078125]], [[51440.71875]], [[47037.28125]], [[53474.1015625]], [[60746.96484375]], [[86316.2734375]], [[50643.44921875]], [[65779.4609375]], [[76064.4296875]], [[57561.7578125]], [[44415.04296875]], [[58072.90625]], [[66083.8046875]], [[58197.02734375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_aacff2447d118a4b78c46019ef4e6497(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8483df49190666ec3f3815f0e624c17f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[52196.23828125]], [[84873.515625]], [[88521.8046875]], [[77473.4140625]], [[56846.40234375]], [[63900.99609375]], [[72542.34375]], [[73128.7890625]], [[36696.59765625]], [[86107.4296875]], [[83204.6796875]], [[41656.7421875]], [[69973.84375]], [[71203.53125]], [[75795.671875]], [[64176.74609375]], [[78790.0625]], [[77350.0703125]], [[75150.515625]], [[71956.296875]], [[78419.7265625]], [[68978.5859375]], [[53145.578125]], [[55659.58203125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_eef5083bbc55e13bb4dbc1b6ccf085e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9759323007ed977717c997c48e97c505
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2d598501cb055fb01ba480b2dbd00e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ab14b2d379857232d90f6495e7412d2
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3778c9b2c69d04a5e888e49321d0cdb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c28fedf580585f5aad19741c7e33b073
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f79ba1cf9740caf15c6e56d9db13b43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0803ba4b5b2be33236c8913b71d383e
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_806016d67805245d5153b9dbe1c9549a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_733c57f1d7d4a060451326b764e29673
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_48e97da5429b3517184cfeec2621f398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_605e6aa58b732b798311b7a1b4f79872(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_43f95a8eea8012075d0edb94600b1446(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22d7ceda27052ee46a0dbc45b6ee442f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88b0ca0685b9342759dfb5d397224017(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef25c8f697eca41640280f31f1bf0aa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1adadb6e94b123888372e975d741351d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89d9a4aba8eb9d3d98af6adc14c976fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2ad4da47d2615ab8a57dc04295e2dfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddf129680ae89a83de59df6b918e9364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ac1ab003834deab946396c4a3f0055(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31fa105003e9e9c40763a53b913698bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0027c0fbf76a57b6e639a394d3c83fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01e9f26064023c39b7adb3c491e5dd4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31fa105003e9e9c40763a53b913698bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdcc307e318128d499e8ce1e6d62a192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ac1ab003834deab946396c4a3f0055(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e021ac22d5a6453bf93a9ffef9006356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e021ac22d5a6453bf93a9ffef9006356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdcc307e318128d499e8ce1e6d62a192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdcc307e318128d499e8ce1e6d62a192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bd9c3eba93baa09aa6648af16235777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.1285748481750488]], [[1.2519538402557373]], [[1.3091199398040771]], [[0.8894222974777222]], [[1.5531479120254517]], [[0.9988418817520142]], [[1.1143465042114258]], [[1.5623719692230225]], [[1.5880999565124512]], [[1.3541173934936523]], [[1.0930718183517456]], [[1.2446036338806152]], [[1.55824613571167]], [[0.9519383907318115]], [[1.7797831296920776]], [[1.385715126991272]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_fdcc307e318128d499e8ce1e6d62a192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a93e667b5210b56f7446aeb9703c746(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88b0ca0685b9342759dfb5d397224017(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31fa105003e9e9c40763a53b913698bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48e97da5429b3517184cfeec2621f398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89d9a4aba8eb9d3d98af6adc14c976fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1adadb6e94b123888372e975d741351d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f409aa8493b0072f6b76a68ecae3beba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1adadb6e94b123888372e975d741351d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82540f4ff7b932aa4b500c6da81204d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48e97da5429b3517184cfeec2621f398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddf129680ae89a83de59df6b918e9364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12e6587de4a27ae5ea89cba38ae163b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aef6ac81bafef093ec79e22eb484197e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31fa105003e9e9c40763a53b913698bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1adadb6e94b123888372e975d741351d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9f6179df67fffcb7b857f584e345141(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ac1ab003834deab946396c4a3f0055(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f79f3763ebdf02b0142e4ea74bef500(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10373b2df4d939df56a0a97bb6d277b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1adadb6e94b123888372e975d741351d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12e6587de4a27ae5ea89cba38ae163b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e021ac22d5a6453bf93a9ffef9006356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_842d7da5ac30af1372b8d6102c5b7a9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1adadb6e94b123888372e975d741351d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2ad4da47d2615ab8a57dc04295e2dfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25e8507847ceb5676bf46d436cc39988(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef25c8f697eca41640280f31f1bf0aa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43f95a8eea8012075d0edb94600b1446(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25e8507847ceb5676bf46d436cc39988(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ac1ab003834deab946396c4a3f0055(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1adadb6e94b123888372e975d741351d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f79f3763ebdf02b0142e4ea74bef500(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01e9f26064023c39b7adb3c491e5dd4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e021ac22d5a6453bf93a9ffef9006356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2ad4da47d2615ab8a57dc04295e2dfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22d7ceda27052ee46a0dbc45b6ee442f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e021ac22d5a6453bf93a9ffef9006356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdcc307e318128d499e8ce1e6d62a192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01e9f26064023c39b7adb3c491e5dd4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddf129680ae89a83de59df6b918e9364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdcc307e318128d499e8ce1e6d62a192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdcc307e318128d499e8ce1e6d62a192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82540f4ff7b932aa4b500c6da81204d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be5991a2fdca9fbbf178e985da73b098(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e021ac22d5a6453bf93a9ffef9006356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ac1ab003834deab946396c4a3f0055(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef25c8f697eca41640280f31f1bf0aa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1adadb6e94b123888372e975d741351d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d4086ffc60fa827190f6fcc412992fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22d7ceda27052ee46a0dbc45b6ee442f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ac1ab003834deab946396c4a3f0055(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01e9f26064023c39b7adb3c491e5dd4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d07426e53adabbe43d725c9182362066(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88b0ca0685b9342759dfb5d397224017(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e021ac22d5a6453bf93a9ffef9006356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebbfecd432611d08ccbd974d4f2bf164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_842d7da5ac30af1372b8d6102c5b7a9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01e9f26064023c39b7adb3c491e5dd4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22d7ceda27052ee46a0dbc45b6ee442f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e021ac22d5a6453bf93a9ffef9006356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da5cac53e8de444d5aa78c9ebe042c3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.0142029523849487]], [[2.053401470184326]], [[1.3052400350570679]], [[1.1951097249984741]], [[2.430665969848633]], [[1.4276739358901978]], [[1.5479648113250732]], [[1.6883649826049805]], [[2.279353618621826]], [[1.985144853591919]], [[1.8742430210113525]], [[1.6576496362686157]], [[1.9620004892349243]], [[1.9623486995697021]], [[1.26690673828125]], [[1.4913731813430786]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_fdcc307e318128d499e8ce1e6d62a192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d07426e53adabbe43d725c9182362066(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9f6179df67fffcb7b857f584e345141(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5f5dd958f14617398be8d6c51dd03e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f79f3763ebdf02b0142e4ea74bef500(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f187b638700f9c890d305e7f8719280a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48e97da5429b3517184cfeec2621f398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdcc307e318128d499e8ce1e6d62a192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1adadb6e94b123888372e975d741351d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10373b2df4d939df56a0a97bb6d277b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a46f6e4e1ef710b7d88d9138361cb41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e021ac22d5a6453bf93a9ffef9006356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22d7ceda27052ee46a0dbc45b6ee442f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10373b2df4d939df56a0a97bb6d277b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a93e667b5210b56f7446aeb9703c746(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_842d7da5ac30af1372b8d6102c5b7a9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5f5dd958f14617398be8d6c51dd03e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c8c6ea296efd3839e12575f5bd44b27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ac1ab003834deab946396c4a3f0055(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48e97da5429b3517184cfeec2621f398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be5991a2fdca9fbbf178e985da73b098(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88b0ca0685b9342759dfb5d397224017(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10373b2df4d939df56a0a97bb6d277b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ac1ab003834deab946396c4a3f0055(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be6bff74a23eeedb7bd280a8c3c7ba9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be6bff74a23eeedb7bd280a8c3c7ba9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be6bff74a23eeedb7bd280a8c3c7ba9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be6bff74a23eeedb7bd280a8c3c7ba9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaa483d9006fbf1cc6455c2655c7effc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaa483d9006fbf1cc6455c2655c7effc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaa483d9006fbf1cc6455c2655c7effc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaa483d9006fbf1cc6455c2655c7effc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_842d7da5ac30af1372b8d6102c5b7a9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdcc307e318128d499e8ce1e6d62a192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89d9a4aba8eb9d3d98af6adc14c976fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89d9a4aba8eb9d3d98af6adc14c976fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89d9a4aba8eb9d3d98af6adc14c976fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e0b4e8c8937855bb08f50f5cac8f9d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d07426e53adabbe43d725c9182362066(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ac1ab003834deab946396c4a3f0055(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89d9a4aba8eb9d3d98af6adc14c976fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22d7ceda27052ee46a0dbc45b6ee442f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22d7ceda27052ee46a0dbc45b6ee442f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aef6ac81bafef093ec79e22eb484197e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f79f3763ebdf02b0142e4ea74bef500(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ce374c580e1202c5fac33e14193c638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ac1ab003834deab946396c4a3f0055(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef25c8f697eca41640280f31f1bf0aa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d4086ffc60fa827190f6fcc412992fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22d7ceda27052ee46a0dbc45b6ee442f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2ad4da47d2615ab8a57dc04295e2dfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef25c8f697eca41640280f31f1bf0aa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdcc307e318128d499e8ce1e6d62a192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdcc307e318128d499e8ce1e6d62a192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdcc307e318128d499e8ce1e6d62a192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e0b4e8c8937855bb08f50f5cac8f9d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e021ac22d5a6453bf93a9ffef9006356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12e6587de4a27ae5ea89cba38ae163b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e021ac22d5a6453bf93a9ffef9006356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_842d7da5ac30af1372b8d6102c5b7a9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ac1ab003834deab946396c4a3f0055(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e021ac22d5a6453bf93a9ffef9006356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a5ba53606939402196a3ef065310960(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ac1ab003834deab946396c4a3f0055(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_842d7da5ac30af1372b8d6102c5b7a9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01e9f26064023c39b7adb3c491e5dd4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d4086ffc60fa827190f6fcc412992fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1adadb6e94b123888372e975d741351d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1adadb6e94b123888372e975d741351d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f79f3763ebdf02b0142e4ea74bef500(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89d9a4aba8eb9d3d98af6adc14c976fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88b0ca0685b9342759dfb5d397224017(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22d7ceda27052ee46a0dbc45b6ee442f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f409aa8493b0072f6b76a68ecae3beba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82540f4ff7b932aa4b500c6da81204d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e759b13ecf318e1857c23178c0b056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a46f6e4e1ef710b7d88d9138361cb41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be5991a2fdca9fbbf178e985da73b098(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ac1ab003834deab946396c4a3f0055(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c8c6ea296efd3839e12575f5bd44b27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25e8507847ceb5676bf46d436cc39988(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10373b2df4d939df56a0a97bb6d277b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12e6587de4a27ae5ea89cba38ae163b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89d9a4aba8eb9d3d98af6adc14c976fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31fa105003e9e9c40763a53b913698bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdcc307e318128d499e8ce1e6d62a192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88b0ca0685b9342759dfb5d397224017(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e759b13ecf318e1857c23178c0b056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2ad4da47d2615ab8a57dc04295e2dfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f187b638700f9c890d305e7f8719280a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdcc307e318128d499e8ce1e6d62a192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddf129680ae89a83de59df6b918e9364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e0b4e8c8937855bb08f50f5cac8f9d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5f5dd958f14617398be8d6c51dd03e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e021ac22d5a6453bf93a9ffef9006356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e021ac22d5a6453bf93a9ffef9006356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aef6ac81bafef093ec79e22eb484197e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22d7ceda27052ee46a0dbc45b6ee442f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_842d7da5ac30af1372b8d6102c5b7a9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88b0ca0685b9342759dfb5d397224017(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12e6587de4a27ae5ea89cba38ae163b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82540f4ff7b932aa4b500c6da81204d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88b0ca0685b9342759dfb5d397224017(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_842d7da5ac30af1372b8d6102c5b7a9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e021ac22d5a6453bf93a9ffef9006356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0027c0fbf76a57b6e639a394d3c83fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0027c0fbf76a57b6e639a394d3c83fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0027c0fbf76a57b6e639a394d3c83fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0027c0fbf76a57b6e639a394d3c83fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4ef02148be47d6e321ff7f589e6b6c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[44111.7109375]], [[17871.0390625]], [[34456.4765625]], [[44836.67578125]], [[48451.37109375]], [[51628.41796875]], [[65512.44921875]], [[63779.671875]], [[57183.37890625]], [[47827.90234375]], [[47769.14453125]], [[50280.84765625]], [[47759.7578125]], [[60607.34375]], [[44385.828125]], [[49698.08984375]], [[52658.55078125]], [[57011.4609375]], [[16017.5703125]], [[56370.58984375]], [[56943.859375]], [[51527.203125]], [[46666.2734375]], [[48208.17578125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_551428fbc14297adcfeb03a38d8a769f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[27788.734375]], [[65264.19921875]], [[60400.171875]], [[49778.5078125]], [[65997.859375]], [[52837.23828125]], [[76932.9453125]], [[41565.7890625]], [[35542.8125]], [[53729.21484375]], [[76907.5703125]], [[65419.49609375]], [[47146.91015625]], [[61715.30078125]], [[50721.5625]], [[73368.6953125]], [[67275.5]], [[31351.744140625]], [[57100.6953125]], [[44081.046875]], [[52843.109375]], [[40803.4921875]], [[42584.96875]], [[77103.3359375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_35da420d71be42827b5300cdec90bc59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[55397.109375]], [[56351.0078125]], [[43602.48828125]], [[90683.5078125]], [[51676.99609375]], [[58209.51171875]], [[62796.50390625]], [[49609.9140625]], [[62965.30078125]], [[62313.9609375]], [[44644.30078125]], [[51440.71875]], [[47037.28125]], [[53474.1015625]], [[60746.96484375]], [[86316.2734375]], [[50643.44921875]], [[65779.4609375]], [[76064.4296875]], [[57561.7578125]], [[44415.04296875]], [[58072.90625]], [[66083.8046875]], [[58197.02734375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_656850cf7ffedb465b7f5231f4c32456(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[52196.23828125]], [[84873.515625]], [[88521.8046875]], [[77473.4140625]], [[56846.40234375]], [[63900.99609375]], [[72542.34375]], [[73128.7890625]], [[36696.59765625]], [[86107.4296875]], [[83204.6796875]], [[41656.7421875]], [[69973.84375]], [[71203.53125]], [[75795.671875]], [[64176.74609375]], [[78790.0625]], [[77350.0703125]], [[75150.515625]], [[71956.296875]], [[78419.7265625]], [[68978.5859375]], [[53145.578125]], [[55659.58203125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_31fa105003e9e9c40763a53b913698bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ce374c580e1202c5fac33e14193c638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0027c0fbf76a57b6e639a394d3c83fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2ad4da47d2615ab8a57dc04295e2dfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8728f2adcf357a45a5f54fb8ff8f9e82
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1adadb6e94b123888372e975d741351d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605e6aa58b732b798311b7a1b4f79872
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()