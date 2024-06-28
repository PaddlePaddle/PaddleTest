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
    class PrimitiveOp_b22e17be2ab0136fb46b2063a7bae3f3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.one_hot(input_0 % input_1, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bae3303a676a0dbd8611f0fde345914c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b22e17be2ab0136fb46b2063a7bae3f3
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 4, 1, 5], dtype='int32').reshape([4]),
                paddle.to_tensor([80], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_89f277238e3de582669352c7d929a687(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.one_hot(input_0 % input_1, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2100], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b951e7f3f887a0886364d79e7fc6087a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89f277238e3de582669352c7d929a687
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.to_tensor([21], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d70188c428eb202544c280d1e2429b59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b22e17be2ab0136fb46b2063a7bae3f3
        def get_inputs(self):
            return [
                paddle.to_tensor([4, 0, 3], dtype='int32').reshape([3]),
                paddle.to_tensor([80], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_fc4cb8dcf3f74be81b5a2ef3ccc50dcb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.one_hot(input_0 % input_1, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3549], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_870fdfa845b321c2f7a0ec8896a7d644(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc4cb8dcf3f74be81b5a2ef3ccc50dcb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([81], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_f9a584a2c359a29afda8a499f0fa66e2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.one_hot(input_0 % input_1, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4116], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c88706a8acd01c0e7a876057c3108d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9a584a2c359a29afda8a499f0fa66e2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([21], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_277a8e627448e957f1cab8b05b77c18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b22e17be2ab0136fb46b2063a7bae3f3
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 3, 5, 3, 5, 6], dtype='int32').reshape([6]),
                paddle.to_tensor([80], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_309de41bc5561eefbb5a93bd90c4d267(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b22e17be2ab0136fb46b2063a7bae3f3
        def get_inputs(self):
            return [
                paddle.to_tensor([9, 1], dtype='int32').reshape([2]),
                paddle.to_tensor([80], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_58474b23aa16bf6e67803b0407657b67(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.one_hot(input_0 % input_1, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1e2b58619f49f552ea80889c8171f368(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58474b23aa16bf6e67803b0407657b67
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 4, 1, 5], dtype='int32').reshape([4]),
                paddle.to_tensor([80], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_6d9e1b9499cb15859b60018e31b1bb90(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.one_hot(input_0 % input_1, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e82fe8f9856a260867696ed5d42cba0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d9e1b9499cb15859b60018e31b1bb90
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.to_tensor([21], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_5a3a794a4f7ddca40a7053f2200c0e50(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.one_hot(input_0 % input_1, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc9479e093ed036e4012f256c0c1cd49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a3a794a4f7ddca40a7053f2200c0e50
        def get_inputs(self):
            return [
                paddle.to_tensor([4, 0, 3], dtype='int32').reshape([3]),
                paddle.to_tensor([80], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_5596ca484a39ceddaccfbb6af7333fd8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.one_hot(input_0 % input_1, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c546860921e995102f0b596dc49ff5a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5596ca484a39ceddaccfbb6af7333fd8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([81], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_60309f2a2193ad06d129cb881366fb69(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.one_hot(input_0 % input_1, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_233f2175b0364cdf006ae2fd0e19b05a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60309f2a2193ad06d129cb881366fb69
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([21], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_afb74e3f8e68c7fa528eb19ef051142c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.one_hot(input_0 % input_1, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8faaba6b7236bf22d84c2713c4e68c00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afb74e3f8e68c7fa528eb19ef051142c
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 3, 5, 3, 5, 6], dtype='int32').reshape([6]),
                paddle.to_tensor([80], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_baf5ea55c533cd9be13f377073b95b44(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.one_hot(input_0 % input_1, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ef4b3415ac179f86a1fe53a7466f7e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_baf5ea55c533cd9be13f377073b95b44
        def get_inputs(self):
            return [
                paddle.to_tensor([9, 1], dtype='int32').reshape([2]),
                paddle.to_tensor([80], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_8160445ea0f29490fe0b928dd8a9fb7a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.one_hot(input_0 % input_1, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_59f3cf94ceb234610c1c3c2834f8b5a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8160445ea0f29490fe0b928dd8a9fb7a
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 4, 1, 5], dtype='int32').reshape([4]),
                paddle.to_tensor([80], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_b5c3282fe802525e9aa33ab32d3b2ca1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.one_hot(input_0 % input_1, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f661b9b98e660e0d885e16c417f501ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5c3282fe802525e9aa33ab32d3b2ca1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.to_tensor([21], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8e7addf1d65308832a1ee4e0f1df94df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8160445ea0f29490fe0b928dd8a9fb7a
        def get_inputs(self):
            return [
                paddle.to_tensor([4, 0, 3], dtype='int32').reshape([3]),
                paddle.to_tensor([80], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2dee917d2c2cabd2426be3d7fe7c7147(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5c3282fe802525e9aa33ab32d3b2ca1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([81], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f52fad929d53d0ea058d8ad2bbf26ad6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5c3282fe802525e9aa33ab32d3b2ca1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([21], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4d5072f84c34d140c15a44a9e592cea7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8160445ea0f29490fe0b928dd8a9fb7a
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 3, 5, 3, 5, 6], dtype='int32').reshape([6]),
                paddle.to_tensor([80], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_fd3bd85ee2d1bd1b5d7b80e1a0088daf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8160445ea0f29490fe0b928dd8a9fb7a
        def get_inputs(self):
            return [
                paddle.to_tensor([9, 1], dtype='int32').reshape([2]),
                paddle.to_tensor([80], dtype='int32').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()