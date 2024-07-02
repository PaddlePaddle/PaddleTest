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
    class PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c72d2d994dc9926ec4769eed9c5d5337(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c551909f3e1a2d2008acfdb0ad77a6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f41e29648b4c31c170bcc63d6cbe0087(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8606d97ab173625d6a572b1aba56b78c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f41e29648b4c31c170bcc63d6cbe0087
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24569982290267944]]], dtype='float32').reshape([1, 1, 1]),
            ]


    
    class PrimitiveOp_7fce434d86f570d5e6b4066d59161ff5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[12096, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[12096, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4bed8165ea8c81faf87173855f9913e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fce434d86f570d5e6b4066d59161ff5
        def get_inputs(self):
            return [
                paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([12096, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cfe605339f1200caf169b8e3f0a348b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ce015e3bbdb57f6222f48fe8b3fd4d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_94c0414b6b2d80e50d88e4fde2fa45f8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6f5b6f3d2d1108de1db80e94420061f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c0414b6b2d80e50d88e4fde2fa45f8
        def get_inputs(self):
            return [
                paddle.to_tensor([1074.00341796875], dtype='float32').reshape([1]),
                paddle.to_tensor(8732.0, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_6a291553382966d0f67f76cb48b561df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c6a26c39017100efb10e089326edfeac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a291553382966d0f67f76cb48b561df
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.00010449004184920341], [2.6468989744898863e-05], [0.0014538551913574338], [0.021787527948617935], [0.005431822966784239], [0.00016644540301058441]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_ae79941b8fef3dd061b9ace75133da02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a291553382966d0f67f76cb48b561df
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[2.79340565612074e-05], [0.0038048378191888332], [0.002504590665921569], [0.0007974102045409381], [0.0007925943937152624], [0.00013527194096241146]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_6884f0d5bd784be9261d177fd093b14c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f41e29648b4c31c170bcc63d6cbe0087
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.012315968982875347], [0.08602733910083771], [0.0671916976571083], [0.09544733166694641], [0.06546390801668167], [0.011264831759035587]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_b6f5e30c3a3154a3b83ccee24bc14437(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2b10a20e0b1970303b671ad304c281f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c72d2d994dc9926ec4769eed9c5d5337(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27f5b7ca811c61b69d3271083cabf7c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_df9f796dda8a6ffb7c5e643551d20320(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(9.278481483459473, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fbb6a0599de43c8daf7115abe2096064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(2.71030592918396, dtype='float32').reshape([]),
                paddle.to_tensor([2.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3afa36369367947cafb3d9433b54033b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f57d9cabaa7da950174491909b0fd0ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f57d9cabaa7da950174491909b0fd0ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d363939b9b7b38a91c4ac0875654969b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(105724.5390625, dtype='float32').reshape([]),
                paddle.to_tensor([0.11950204521417618], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ce83c0d75a0f74ef9b0b962ec3f6d13a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(103881.3828125, dtype='float32').reshape([]),
                paddle.to_tensor([0.11950204521417618], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d53f08a21bf22d2044cd25df2be45b96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(962.5842895507812, dtype='float32').reshape([]),
                paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_24b25d457e390884b5c26588a80de4a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dfb93929a947ff1efb6983702023b4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e05f6b81a1646fe612585d40d93d340e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5376, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[5376, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7d939a1b5e29e5aea4955fe28ab7c930(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e05f6b81a1646fe612585d40d93d340e
        def get_inputs(self):
            return [
                paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([5376, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72687fd503893d28d9aa4befb426c2f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ddbcd11f29a70881c19fd5c8b328606(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2e687b0c95488e161452102794d164e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.006484865676611662], [0.05156746506690979], [-0.11047624051570892], [0.0472310408949852], [-0.024203266948461533], [0.03868870064616203], [-0.07902292162179947], [0.15735942125320435], [0.08276878297328949]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_20fa68edd9a25a60737361d33d757aea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.01888887956738472], [-0.020349886268377304], [0.06796149909496307], [-0.025028301402926445], [0.1351655125617981], [0.03656329587101936], [0.11209447681903839], [-0.09510591626167297], [0.012649431824684143]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.02537374570965767], [0.031217578798532486], [-0.04251473769545555], [0.022202739492058754], [0.11096224188804626], [0.0752519965171814], [0.03307155892252922], [0.06225350871682167], [0.09541821479797363]], dtype='float32').reshape([9, 1]),
            ]


    
    class PrimitiveOp_55f237ad98ee678061b8a243a3ad6353(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_757b3d73e415b233eebfb80e332defd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f237ad98ee678061b8a243a3ad6353
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.37376904487609863], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3ecda7e2e19dc0a9cb632f202b7aaaa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ecda7e2e19dc0a9cb632f202b7aaaa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0d41d11f0e3b4bef9646e7d307512ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(16057.021484375, dtype='float32').reshape([]),
                paddle.to_tensor([0.10695494711399078], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f884f1d5026ed263944ab6839c9b12cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(3966.4541015625, dtype='float32').reshape([]),
                paddle.to_tensor([0.10695494711399078], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_2e2af5ef8bbbc99d746b9e561189c5ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_506593ee859c2c3748bcd4f4c9dba617(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e2af5ef8bbbc99d746b9e561189c5ec
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.0, -0.0, 0.0, 0.0, -0.0, -0.0], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.02505781129002571, 0.017090944573283195, 0.004393713548779488, -0.000812494894489646, 0.017373912036418915, 0.0037726969458162785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_99f5d4c9e1e6c23080b269533bd39350(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e2af5ef8bbbc99d746b9e561189c5ec
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0031286091543734074, 0.023960445076227188, 0.052997514605522156, 0.13812905550003052, 0.009350953623652458, 0.12514647841453552], dtype='float32').reshape([6]),
                paddle.to_tensor([0.05580516159534454, 0.21465319395065308, 0.09616874158382416, 0.3242293894290924, 0.04169569909572601, 0.15700317919254303], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0cb03dec2d71fccebed81fc45d87f09a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e2af5ef8bbbc99d746b9e561189c5ec
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.2766314744949341, 0.10538440942764282, 0.14060714840888977, -0.007649481296539307, 0.0995698869228363, 0.10518306493759155], dtype='float32').reshape([6]),
                paddle.to_tensor([0.09058192372322083, 0.1621771603822708, 0.031248152256011963, 0.106215700507164, 0.17448961734771729, 0.0300922691822052], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_abc4d93d09988d7576426df34a0a7246(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e2af5ef8bbbc99d746b9e561189c5ec
        def get_inputs(self):
            return [
                paddle.to_tensor([0.21817441284656525, -0.05455118417739868, -0.2314678281545639, 0.05182693898677826, -0.13170656561851501, 0.02061089128255844], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.051661550998687744, 0.42682701349258423, 0.020339012145996094, -0.39126476645469666, -0.16860561072826385, 0.02947470359504223], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_f6dcc1815bd4715c4adaa30735aaa59a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e2af5ef8bbbc99d746b9e561189c5ec
        def get_inputs(self):
            return [
                paddle.to_tensor([0.002855276456102729, 0.20050019025802612, 3.2579751014709473, 0.0014492734335362911, 0.00847349502146244, 0.18845637142658234], dtype='float32').reshape([6]),
                paddle.to_tensor([1.0028553009033203, 1.200500249862671, 4.257975101470947, 1.0014492273330688, 1.008473515510559, 1.188456416130066], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_cb1a61252eed6d534018a7ae697e952e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb1a61252eed6d534018a7ae697e952e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ae4a1189211b26ce8436f7eb9859280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(-3069311.75, dtype='float32').reshape([]),
                paddle.to_tensor([0.31719982624053955], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c7b51f5e651eecae5d556055f80b98eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(106358.171875, dtype='float32').reshape([]),
                paddle.to_tensor([0.31719982624053955], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a8ba00aaac96e425dbeeb95a7be851e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(941.97998046875, dtype='float32').reshape([]),
                paddle.to_tensor([4.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_c58058167ce95f300ff7001a51747a2a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8400, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[8400, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1de8f9ddec8946b8aad7b0d673f69fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c58058167ce95f300ff7001a51747a2a
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([8400, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6651b57381524647045dc4bcf7590ebf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41566054c51a2c332d62f75009af1464(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6651b57381524647045dc4bcf7590ebf
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ce015e3bbdb57f6222f48fe8b3fd4d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dfb93929a947ff1efb6983702023b4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b533720bd38e7d0a7d050dffef1a610(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b533720bd38e7d0a7d050dffef1a610(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28c7031c879bd182a41917ae917fb41d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(187351.796875, dtype='float32').reshape([]),
                paddle.to_tensor([0.1571148782968521], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8affabcedf2704e59c1585e2d08c55ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(85683.6015625, dtype='float32').reshape([]),
                paddle.to_tensor([0.1571148782968521], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7a4ec435f2047515196605808713b076(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f41e29648b4c31c170bcc63d6cbe0087
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.2439064234495163], [0.24398155510425568]]], dtype='float32').reshape([1, 2, 1]),
            ]


    class TestPrimitiveOp_e2b10a20e0b1970303b671ad304c281f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c551909f3e1a2d2008acfdb0ad77a6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_400dc6d2ee70174e8d90ec203f9d3dfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.01941882260143757]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ae84ba83a1d4aed540bef6c73c893747(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.006170423701405525]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.025589246302843094]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_0c55dff143f08d76d7767c6d2e831028(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-0.043292563408613205], [-0.02788546308875084], [0.05995785444974899], [0.03143922984600067], [-0.0009129985119216144], [-0.034111231565475464]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_4e1eeac6c168ce4736aa0dd5fb1f2746(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.045713815838098526], [0.040647849440574646], [0.06487330794334412], [-0.023683693259954453], [0.05853608250617981], [0.06354904174804688]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.002421251032501459], [0.012762386351823807], [0.12483116239309311], [0.007755537051707506], [0.057623084634542465], [0.02943781390786171]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_e1f563d92c66804790cc1c131445135c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f41e29648b4c31c170bcc63d6cbe0087
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24656285345554352]]], dtype='float32').reshape([1, 1, 1]),
            ]


    
    class PrimitiveOp_9338f2fe61b0d54b9e549be33ff58a0b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c2ad7659088dc2ed5f4648dd557123c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9338f2fe61b0d54b9e549be33ff58a0b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24b25d457e390884b5c26588a80de4a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cfe605339f1200caf169b8e3f0a348b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa75dcca0f13d88150349f91d3c36777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(56.84940719604492, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_39e8186af91eacef386cf8885e0a3d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(547.307373046875, dtype='float32').reshape([]),
                paddle.to_tensor([4.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_52fddc906e7c5b9a428e35ad2113fc67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_52fddc906e7c5b9a428e35ad2113fc67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f77f15ea0f78683a5630e4a0b0742e0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(479493.6875, dtype='float32').reshape([]),
                paddle.to_tensor([0.32257935404777527], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c9cf028356f60ca8879e2d19b5490091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(118053.0625, dtype='float32').reshape([]),
                paddle.to_tensor([0.32257935404777527], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_28b54aa12978a233e012db6265f3fd3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9338f2fe61b0d54b9e549be33ff58a0b
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3c521d59fc4670b5255a4485accbf18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f68ec8c02316e2f3a922c7f01f3e5125(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f237ad98ee678061b8a243a3ad6353
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.19904105365276337], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fb68d9f2fc3baf9eaec1fed989878a63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb68d9f2fc3baf9eaec1fed989878a63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8695696bea73401952193944084db64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(-3186.49609375, dtype='float32').reshape([]),
                paddle.to_tensor([0.3305808901786804], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cde8f91a0173dfc42fa6399b63507ff3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(262612.125, dtype='float32').reshape([]),
                paddle.to_tensor([0.3305808901786804], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_caf985e91b7610c665f03be99339e975(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c0414b6b2d80e50d88e4fde2fa45f8
        def get_inputs(self):
            return [
                paddle.to_tensor([308.5841369628906], dtype='float32').reshape([1]),
                paddle.to_tensor(2434.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_f67988f2a6dd56663853f4fe1bcdcd15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f67988f2a6dd56663853f4fe1bcdcd15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6429d51d30c04660f34161ad2e61486c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(1073.373046875, dtype='float32').reshape([]),
                paddle.to_tensor([0.2833605110645294], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_99195c9a5477f68c0592272713307bee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(15778.453125, dtype='float32').reshape([]),
                paddle.to_tensor([0.2833605110645294], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b6f5e30c3a3154a3b83ccee24bc14437(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_29698d43d9010b0bf9cc8880283013a5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[100, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_02fdc4cde64080b14081c80861f38238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29698d43d9010b0bf9cc8880283013a5
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6cf0009ba54535ba39c68c810c110f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.050307195633649826, 0.24958865344524384, 0.12819220125675201, 0.15536218881607056], [0.31391292810440063, 0.12240537256002426, 0.3002643585205078, 0.22792692482471466]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.08483082801103592, 0.046015478670597076, 0.15802784264087677, 0.42296135425567627], [0.37927430868148804, 0.43439051508903503, 0.4780726134777069, 0.20804838836193085]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_72687fd503893d28d9aa4befb426c2f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e982a96d691bf2669cf7b3b506caeffd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6069, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[6069, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e00deb4134e1421dbecfb15603b0414(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e982a96d691bf2669cf7b3b506caeffd
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([6069, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_97648ab39fa378a8135d61477e6520cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[300, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_915d60dc0575330f191b9d45d7de71cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97648ab39fa378a8135d61477e6520cd
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65327f0963c635dd3bdde0b5acb43bf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15910319983959198, 0.37697386741638184, 0.07013243436813354, 0.14398092031478882], [0.3346053659915924, 0.4726791977882385, 0.03578709438443184, 0.40225374698638916]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.3335762619972229, 0.255480021238327, 0.28426453471183777, 0.41945880651474], [0.26453742384910583, 0.30485397577285767, 0.11422444134950638, 0.10182178020477295]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_07c8faad31c3cf883fb016032a7e28d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.01143362931907177], [0.00884125754237175], [0.06832607835531235], [0.0002330933784833178], [0.1191963478922844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_7e38a83a06f171ec593de2ef9d34799d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.006949621252715588], [-0.07074487954378128], [-0.06696217507123947], [0.05739561840891838], [-0.10817878693342209]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.004484008066356182], [-0.06190362200140953], [0.001363899908028543], [0.05762871354818344], [0.01101756189018488]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_4e43afb7ac5729ef2b1d37fecc3d335b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f237ad98ee678061b8a243a3ad6353
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3146562874317169], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3afa36369367947cafb3d9433b54033b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ddbcd11f29a70881c19fd5c8b328606(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94215ded7f401384c173969ab7271ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca6be1ca7a0eab7acd06f2237dc44495(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca6be1ca7a0eab7acd06f2237dc44495(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f2f66ea1553f571619d7264d7dd0811(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(786341.75, dtype='float32').reshape([]),
                paddle.to_tensor([0.4607619047164917], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_22cdb33cb1605b775556240a6e876922(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(134349.171875, dtype='float32').reshape([]),
                paddle.to_tensor([0.4607619047164917], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4f0c53b627d0737a3febe216318ddb36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f0c53b627d0737a3febe216318ddb36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d043b43660cbe1d7920bca1b379d06fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(313841.75, dtype='float32').reshape([]),
                paddle.to_tensor([0.3013629615306854], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cf16604d897ea2ad2e1a2c3a19333b20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(173847.609375, dtype='float32').reshape([]),
                paddle.to_tensor([0.3013629615306854], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ab4fa4ee1f12261cdd7bde09a9aa9656(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab4fa4ee1f12261cdd7bde09a9aa9656(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a880edf6574eb602644c75ab031f10a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(-143355.796875, dtype='float32').reshape([]),
                paddle.to_tensor([0.2739626169204712], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_417966cef488f8d0862a55ad8c23fd4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(215886.578125, dtype='float32').reshape([]),
                paddle.to_tensor([0.2739626169204712], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_10d99b41822ba79ee0b81acf28640d78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f237ad98ee678061b8a243a3ad6353
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3481174111366272], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d3c521d59fc4670b5255a4485accbf18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dc86cf6f2efa5a551133a8ce0f68d47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(15.708755493164062, dtype='float32').reshape([]),
                paddle.to_tensor([3.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_57fb486656911cd52f8cb055784cef98(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20267, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[20267, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6101f99ff6f0ebd69ccdceebd53bfae0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57fb486656911cd52f8cb055784cef98
        def get_inputs(self):
            return [
                paddle.uniform([20267, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([20267, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6bad7fdaa06fbccb6de619a72fb420e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.043187495321035385], [-0.013158541172742844], [-0.05806963890790939], [0.015963705256581306]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_833e6173a8009b7430841539fb3f032b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0007632412016391754], [-0.003575025126338005], [0.15011842548847198], [-0.005106845870614052]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.04242425411939621], [-0.01673356629908085], [0.09204878658056259], [0.010856859385967255]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_6c2a409b45523ab814fd942ff9a99769(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(4.327864170074463, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2e29941125e94be2abcf7cea757016ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e29941125e94be2abcf7cea757016ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243b2b0efdbb8efabbe389b50db6a990(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(43973.95703125, dtype='float32').reshape([]),
                paddle.to_tensor([0.20547689497470856], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_235d6d00aa7048e03072bdec1a9266e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(29822.849609375, dtype='float32').reshape([]),
                paddle.to_tensor([0.20547689497470856], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3bf5aea36276337a85bbf738352ce9b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f237ad98ee678061b8a243a3ad6353
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.44187647104263306], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_033e0e786e093ec20cd1ed7b2e0dd6a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(33.15422058105469, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_95185a129178a21599c0c7a5c2ba1224(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6804, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[6804, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6ed73e0b152913272297c7f018b5ff9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95185a129178a21599c0c7a5c2ba1224
        def get_inputs(self):
            return [
                paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([6804, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94215ded7f401384c173969ab7271ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1e940be7422c9e6633ffe7c3421025e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(234.3687744140625, dtype='float32').reshape([]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3b859d3a320241fa51cabca98174065d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(137.35140991210938, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_27f5b7ca811c61b69d3271083cabf7c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7f277270e495f181fa076531890efb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d542667e016d38b5d2ee9faa073e7cd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d542667e016d38b5d2ee9faa073e7cd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e230b2fc807dc17f876d6c5091d59d7
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbfae8105663e4bacad472691ec6890a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(14909.78125, dtype='float32').reshape([]),
                paddle.to_tensor([0.31060782074928284], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_72cf4792eacdf204a385a9a9b24d6d4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(242725.671875, dtype='float32').reshape([]),
                paddle.to_tensor([0.31060782074928284], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c7f277270e495f181fa076531890efb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c72d2d994dc9926ec4769eed9c5d5337(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c551909f3e1a2d2008acfdb0ad77a6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f455825aa9c52bf96c36c06fd8c9f447(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a291553382966d0f67f76cb48b561df
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24569982290267944]]], dtype='float32').reshape([1, 1, 1]),
            ]


    class TestPrimitiveOp_d78b815cbdb16c8def1badfe67fab400(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([12096, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cfe605339f1200caf169b8e3f0a348b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ce015e3bbdb57f6222f48fe8b3fd4d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f5b6f3d2d1108de1db80e94420061f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c0414b6b2d80e50d88e4fde2fa45f8
        def get_inputs(self):
            return [
                paddle.to_tensor([1074.00341796875], dtype='float32').reshape([1]),
                paddle.to_tensor(8732.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_c6a26c39017100efb10e089326edfeac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a291553382966d0f67f76cb48b561df
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.00010449004184920341], [2.6468989744898863e-05], [0.0014538551913574338], [0.021787527948617935], [0.005431822966784239], [0.00016644540301058441]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_ae79941b8fef3dd061b9ace75133da02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a291553382966d0f67f76cb48b561df
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[2.79340565612074e-05], [0.0038048378191888332], [0.002504590665921569], [0.0007974102045409381], [0.0007925943937152624], [0.00013527194096241146]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_8a899aaecdd6b4e1f003a93633a1a4a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a291553382966d0f67f76cb48b561df
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.012315968982875347], [0.08602733910083771], [0.0671916976571083], [0.09544733166694641], [0.06546390801668167], [0.011264831759035587]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_b6f5e30c3a3154a3b83ccee24bc14437(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2b10a20e0b1970303b671ad304c281f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c72d2d994dc9926ec4769eed9c5d5337(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27f5b7ca811c61b69d3271083cabf7c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df9f796dda8a6ffb7c5e643551d20320(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(9.278481483459473, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fbb6a0599de43c8daf7115abe2096064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(2.71030592918396, dtype='float32').reshape([]),
                paddle.to_tensor([2.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3afa36369367947cafb3d9433b54033b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_970d169b0de9092a65d7681ab57b6ef4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_970d169b0de9092a65d7681ab57b6ef4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d363939b9b7b38a91c4ac0875654969b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(105724.5390625, dtype='float32').reshape([]),
                paddle.to_tensor([0.11950204521417618], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ce83c0d75a0f74ef9b0b962ec3f6d13a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(103881.3828125, dtype='float32').reshape([]),
                paddle.to_tensor([0.11950204521417618], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d53f08a21bf22d2044cd25df2be45b96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(962.5842895507812, dtype='float32').reshape([]),
                paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_24b25d457e390884b5c26588a80de4a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dfb93929a947ff1efb6983702023b4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3acd297e049d6b5f9d5232824b0572cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([5376, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72687fd503893d28d9aa4befb426c2f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ddbcd11f29a70881c19fd5c8b328606(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e687b0c95488e161452102794d164e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.006484865676611662], [0.05156746506690979], [-0.11047624051570892], [0.0472310408949852], [-0.024203266948461533], [0.03868870064616203], [-0.07902292162179947], [0.15735942125320435], [0.08276878297328949]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_20fa68edd9a25a60737361d33d757aea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.01888887956738472], [-0.020349886268377304], [0.06796149909496307], [-0.025028301402926445], [0.1351655125617981], [0.03656329587101936], [0.11209447681903839], [-0.09510591626167297], [0.012649431824684143]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.02537374570965767], [0.031217578798532486], [-0.04251473769545555], [0.022202739492058754], [0.11096224188804626], [0.0752519965171814], [0.03307155892252922], [0.06225350871682167], [0.09541821479797363]], dtype='float32').reshape([9, 1]),
            ]


    
    class PrimitiveOp_016389a07bc00ec634e2445a5ef6873b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c4d6a837445ca63a891b48a34686a84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_016389a07bc00ec634e2445a5ef6873b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.37376904487609863], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_296dae8b856734ca36cdc5b333dae729(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_296dae8b856734ca36cdc5b333dae729(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0d41d11f0e3b4bef9646e7d307512ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(16057.021484375, dtype='float32').reshape([]),
                paddle.to_tensor([0.10695494711399078], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f884f1d5026ed263944ab6839c9b12cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(3966.4541015625, dtype='float32').reshape([]),
                paddle.to_tensor([0.10695494711399078], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_506593ee859c2c3748bcd4f4c9dba617(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e2af5ef8bbbc99d746b9e561189c5ec
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.0, -0.0, 0.0, 0.0, -0.0, -0.0], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.02505781129002571, 0.017090944573283195, 0.004393713548779488, -0.000812494894489646, 0.017373912036418915, 0.0037726969458162785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_99f5d4c9e1e6c23080b269533bd39350(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e2af5ef8bbbc99d746b9e561189c5ec
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0031286091543734074, 0.023960445076227188, 0.052997514605522156, 0.13812905550003052, 0.009350953623652458, 0.12514647841453552], dtype='float32').reshape([6]),
                paddle.to_tensor([0.05580516159534454, 0.21465319395065308, 0.09616874158382416, 0.3242293894290924, 0.04169569909572601, 0.15700317919254303], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0cb03dec2d71fccebed81fc45d87f09a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e2af5ef8bbbc99d746b9e561189c5ec
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.2766314744949341, 0.10538440942764282, 0.14060714840888977, -0.007649481296539307, 0.0995698869228363, 0.10518306493759155], dtype='float32').reshape([6]),
                paddle.to_tensor([0.09058192372322083, 0.1621771603822708, 0.031248152256011963, 0.106215700507164, 0.17448961734771729, 0.0300922691822052], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_abc4d93d09988d7576426df34a0a7246(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e2af5ef8bbbc99d746b9e561189c5ec
        def get_inputs(self):
            return [
                paddle.to_tensor([0.21817441284656525, -0.05455118417739868, -0.2314678281545639, 0.05182693898677826, -0.13170656561851501, 0.02061089128255844], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.051661550998687744, 0.42682701349258423, 0.020339012145996094, -0.39126476645469666, -0.16860561072826385, 0.02947470359504223], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_f6dcc1815bd4715c4adaa30735aaa59a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e2af5ef8bbbc99d746b9e561189c5ec
        def get_inputs(self):
            return [
                paddle.to_tensor([0.002855276456102729, 0.20050019025802612, 3.2579751014709473, 0.0014492734335362911, 0.00847349502146244, 0.18845637142658234], dtype='float32').reshape([6]),
                paddle.to_tensor([1.0028553009033203, 1.200500249862671, 4.257975101470947, 1.0014492273330688, 1.008473515510559, 1.188456416130066], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_bc32da780457b95fb6cd5e420472ac2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc32da780457b95fb6cd5e420472ac2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ae4a1189211b26ce8436f7eb9859280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(-3069311.75, dtype='float32').reshape([]),
                paddle.to_tensor([0.31719982624053955], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c7b51f5e651eecae5d556055f80b98eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(106358.171875, dtype='float32').reshape([]),
                paddle.to_tensor([0.31719982624053955], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a8ba00aaac96e425dbeeb95a7be851e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(941.97998046875, dtype='float32').reshape([]),
                paddle.to_tensor([4.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_90672944dc2aeba1f38f9b3fd08905df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([8400, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d9e9b7f5a4af830d370e8c9975604157(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e631a983f91fac91db9c882440774128(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9e9b7f5a4af830d370e8c9975604157
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ce015e3bbdb57f6222f48fe8b3fd4d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dfb93929a947ff1efb6983702023b4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcd843ea94718088c5a66d8c84f1b804(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcd843ea94718088c5a66d8c84f1b804(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28c7031c879bd182a41917ae917fb41d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(187351.796875, dtype='float32').reshape([]),
                paddle.to_tensor([0.1571148782968521], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8affabcedf2704e59c1585e2d08c55ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(85683.6015625, dtype='float32').reshape([]),
                paddle.to_tensor([0.1571148782968521], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b22c160f8f2248409aee2253ee47cc06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a291553382966d0f67f76cb48b561df
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.2439064234495163], [0.24398155510425568]]], dtype='float32').reshape([1, 2, 1]),
            ]


    class TestPrimitiveOp_e2b10a20e0b1970303b671ad304c281f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c551909f3e1a2d2008acfdb0ad77a6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_400dc6d2ee70174e8d90ec203f9d3dfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.01941882260143757]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ae84ba83a1d4aed540bef6c73c893747(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.006170423701405525]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.025589246302843094]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_0c55dff143f08d76d7767c6d2e831028(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-0.043292563408613205], [-0.02788546308875084], [0.05995785444974899], [0.03143922984600067], [-0.0009129985119216144], [-0.034111231565475464]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_4e1eeac6c168ce4736aa0dd5fb1f2746(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.045713815838098526], [0.040647849440574646], [0.06487330794334412], [-0.023683693259954453], [0.05853608250617981], [0.06354904174804688]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.002421251032501459], [0.012762386351823807], [0.12483116239309311], [0.007755537051707506], [0.057623084634542465], [0.02943781390786171]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_bc9a3c5c4a0a6bbba6d42df4175985f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a291553382966d0f67f76cb48b561df
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24656285345554352]]], dtype='float32').reshape([1, 1, 1]),
            ]


    class TestPrimitiveOp_aa86e2772f8df9844cc7c7f436766752(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9e9b7f5a4af830d370e8c9975604157
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24b25d457e390884b5c26588a80de4a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cfe605339f1200caf169b8e3f0a348b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa75dcca0f13d88150349f91d3c36777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(56.84940719604492, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_39e8186af91eacef386cf8885e0a3d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(547.307373046875, dtype='float32').reshape([]),
                paddle.to_tensor([4.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_347ddab96b340827ac6ae4b39e46fa9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_347ddab96b340827ac6ae4b39e46fa9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f77f15ea0f78683a5630e4a0b0742e0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(479493.6875, dtype='float32').reshape([]),
                paddle.to_tensor([0.32257935404777527], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c9cf028356f60ca8879e2d19b5490091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(118053.0625, dtype='float32').reshape([]),
                paddle.to_tensor([0.32257935404777527], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3a59f43f59e32b1534adfbb5a82f3148(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9e9b7f5a4af830d370e8c9975604157
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3c521d59fc4670b5255a4485accbf18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef2ddf3e36cbfc5883a16c9f41f6e89b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_016389a07bc00ec634e2445a5ef6873b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.19904105365276337], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0e7248d3d7d997fa13c6c8b169d68eed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e7248d3d7d997fa13c6c8b169d68eed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8695696bea73401952193944084db64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(-3186.49609375, dtype='float32').reshape([]),
                paddle.to_tensor([0.3305808901786804], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cde8f91a0173dfc42fa6399b63507ff3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(262612.125, dtype='float32').reshape([]),
                paddle.to_tensor([0.3305808901786804], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_caf985e91b7610c665f03be99339e975(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c0414b6b2d80e50d88e4fde2fa45f8
        def get_inputs(self):
            return [
                paddle.to_tensor([308.5841369628906], dtype='float32').reshape([1]),
                paddle.to_tensor(2434.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_59532ef9a4d25cddae8d65c89d288d95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59532ef9a4d25cddae8d65c89d288d95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6429d51d30c04660f34161ad2e61486c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(1073.373046875, dtype='float32').reshape([]),
                paddle.to_tensor([0.2833605110645294], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_99195c9a5477f68c0592272713307bee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(15778.453125, dtype='float32').reshape([]),
                paddle.to_tensor([0.2833605110645294], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b6f5e30c3a3154a3b83ccee24bc14437(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa2885f3967fe512ae768eaa65f7f67b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6cf0009ba54535ba39c68c810c110f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.050307195633649826, 0.24958865344524384, 0.12819220125675201, 0.15536218881607056], [0.31391292810440063, 0.12240537256002426, 0.3002643585205078, 0.22792692482471466]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.08483082801103592, 0.046015478670597076, 0.15802784264087677, 0.42296135425567627], [0.37927430868148804, 0.43439051508903503, 0.4780726134777069, 0.20804838836193085]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_72687fd503893d28d9aa4befb426c2f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_015c1fbecfc127787f4c7cb7ec5bab8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([6069, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81e642f3dc1b73115d59db2b08c92fa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65327f0963c635dd3bdde0b5acb43bf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15910319983959198, 0.37697386741638184, 0.07013243436813354, 0.14398092031478882], [0.3346053659915924, 0.4726791977882385, 0.03578709438443184, 0.40225374698638916]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.3335762619972229, 0.255480021238327, 0.28426453471183777, 0.41945880651474], [0.26453742384910583, 0.30485397577285767, 0.11422444134950638, 0.10182178020477295]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_07c8faad31c3cf883fb016032a7e28d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.01143362931907177], [0.00884125754237175], [0.06832607835531235], [0.0002330933784833178], [0.1191963478922844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_7e38a83a06f171ec593de2ef9d34799d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.006949621252715588], [-0.07074487954378128], [-0.06696217507123947], [0.05739561840891838], [-0.10817878693342209]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.004484008066356182], [-0.06190362200140953], [0.001363899908028543], [0.05762871354818344], [0.01101756189018488]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_3fe90504b45db8abdef0c8b6bbd7a1c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_016389a07bc00ec634e2445a5ef6873b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3146562874317169], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3afa36369367947cafb3d9433b54033b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ddbcd11f29a70881c19fd5c8b328606(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94215ded7f401384c173969ab7271ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99ac7328717124378e75f8c08ca41a80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99ac7328717124378e75f8c08ca41a80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f2f66ea1553f571619d7264d7dd0811(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(786341.75, dtype='float32').reshape([]),
                paddle.to_tensor([0.4607619047164917], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_22cdb33cb1605b775556240a6e876922(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(134349.171875, dtype='float32').reshape([]),
                paddle.to_tensor([0.4607619047164917], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4c7cc1a8b79fde9d6368d461c27487ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c7cc1a8b79fde9d6368d461c27487ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d043b43660cbe1d7920bca1b379d06fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(313841.75, dtype='float32').reshape([]),
                paddle.to_tensor([0.3013629615306854], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cf16604d897ea2ad2e1a2c3a19333b20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(173847.609375, dtype='float32').reshape([]),
                paddle.to_tensor([0.3013629615306854], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a0f9592b3d93a71a23c6058adbdf6acd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0f9592b3d93a71a23c6058adbdf6acd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a880edf6574eb602644c75ab031f10a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(-143355.796875, dtype='float32').reshape([]),
                paddle.to_tensor([0.2739626169204712], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_417966cef488f8d0862a55ad8c23fd4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(215886.578125, dtype='float32').reshape([]),
                paddle.to_tensor([0.2739626169204712], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_343e2bbe1a8c797d725c3be06c87372a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_016389a07bc00ec634e2445a5ef6873b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3481174111366272], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d3c521d59fc4670b5255a4485accbf18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dc86cf6f2efa5a551133a8ce0f68d47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(15.708755493164062, dtype='float32').reshape([]),
                paddle.to_tensor([3.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_96c7f502332b85e231e5c8612ff5427d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([20267, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([20267, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6bad7fdaa06fbccb6de619a72fb420e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.043187495321035385], [-0.013158541172742844], [-0.05806963890790939], [0.015963705256581306]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_833e6173a8009b7430841539fb3f032b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0007632412016391754], [-0.003575025126338005], [0.15011842548847198], [-0.005106845870614052]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.04242425411939621], [-0.01673356629908085], [0.09204878658056259], [0.010856859385967255]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_6c2a409b45523ab814fd942ff9a99769(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(4.327864170074463, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1db5cc321738295010289318c2b2f3ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1db5cc321738295010289318c2b2f3ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243b2b0efdbb8efabbe389b50db6a990(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(43973.95703125, dtype='float32').reshape([]),
                paddle.to_tensor([0.20547689497470856], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_235d6d00aa7048e03072bdec1a9266e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(29822.849609375, dtype='float32').reshape([]),
                paddle.to_tensor([0.20547689497470856], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f41a669ff4226ef468a3db477cac428d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_016389a07bc00ec634e2445a5ef6873b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.44187647104263306], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_033e0e786e093ec20cd1ed7b2e0dd6a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(33.15422058105469, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7541966198fda62dc37b71cd138320d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([6804, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94215ded7f401384c173969ab7271ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1e940be7422c9e6633ffe7c3421025e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(234.3687744140625, dtype='float32').reshape([]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3b859d3a320241fa51cabca98174065d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(137.35140991210938, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_27f5b7ca811c61b69d3271083cabf7c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7f277270e495f181fa076531890efb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4388f318e1670072a297767d4b9bf9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4388f318e1670072a297767d4b9bf9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e530a8b656ff7d0df38deb368c9ca6d3
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbfae8105663e4bacad472691ec6890a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(14909.78125, dtype='float32').reshape([]),
                paddle.to_tensor([0.31060782074928284], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_72cf4792eacdf204a385a9a9b24d6d4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a618e448810c1edc50dbc76730f0fd3a
        def get_inputs(self):
            return [
                paddle.to_tensor(242725.671875, dtype='float32').reshape([]),
                paddle.to_tensor([0.31060782074928284], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c7f277270e495f181fa076531890efb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69a0feb010bac15cc6697594ff6a2fc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()