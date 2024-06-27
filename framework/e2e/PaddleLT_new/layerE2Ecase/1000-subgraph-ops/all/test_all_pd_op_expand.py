import os
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
    class PrimitiveOp_8142ee934be124a8e6b808068b06d2db(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 24, 36]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_706a21720fef2ee13d6c35b67bf21c07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8142ee934be124a8e6b808068b06d2db
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 24, 36], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_706a21720fef2ee13d6c35b67bf21c07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8142ee934be124a8e6b808068b06d2db
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 24, 36], dtype='int64').reshape([4]),
            ]


    
    class PrimitiveOp_33aacc84707d04092c21218cf349a602(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, -1, -1]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 150, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d6a0d913dbbc59fc643658bd710a121a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33aacc84707d04092c21218cf349a602
        def get_inputs(self):
            return [
                paddle.uniform([1, 150, 384], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_51c6adaadb5fdea82300c359f98c254b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 256, 21]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f177eaa96ed1fdb480b70a0319008ee4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51c6adaadb5fdea82300c359f98c254b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 256, 21], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_5627d769d15b86bf6b16a5319152a472(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 25, 38]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9d54486553067c43a7951a9667772a38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5627d769d15b86bf6b16a5319152a472
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 25, 38], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_9d54486553067c43a7951a9667772a38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5627d769d15b86bf6b16a5319152a472
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 25, 38], dtype='int64').reshape([4]),
            ]


    
    class PrimitiveOp_ea8cc6904c5fa2b888f0ee696f0e83ac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 20, 30]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea1f038451850c6bf3fed2edec3a135f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea8cc6904c5fa2b888f0ee696f0e83ac
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 20, 30], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_ea1f038451850c6bf3fed2edec3a135f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea8cc6904c5fa2b888f0ee696f0e83ac
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 20, 30], dtype='int64').reshape([4]),
            ]


    
    class PrimitiveOp_c5dd9d14cc26b0a8447bf0818b5cef84(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, -1, -1]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9126bd6a1495c02150b89aaed372cec2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5dd9d14cc26b0a8447bf0818b5cef84
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 768], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_e618b502961691f2e4af4c505b33d0ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [7, 256, 19]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b90e559c5ebde12e1fa5b9dd59084b37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e618b502961691f2e4af4c505b33d0ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 256, 19], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_2987fa7a2fa8efecede472482b38b717(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 15, 25]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e54dc754334542819b599e79bb2eec20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2987fa7a2fa8efecede472482b38b717
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 15, 25], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_e54dc754334542819b599e79bb2eec20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2987fa7a2fa8efecede472482b38b717
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 15, 25], dtype='int64').reshape([4]),
            ]


    
    class PrimitiveOp_84e352f55ad21d125e47ade527c7a776(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, -1, -1]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 150, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5d3b5e692c03b8a0af7231db97038d6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84e352f55ad21d125e47ade527c7a776
        def get_inputs(self):
            return [
                paddle.uniform([1, 150, 768], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_e79513d94bcad4b0fbc8f3bac306cd1c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [8, 256, 150]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ef584a52bd9e070d54ed7d5d47af573e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e79513d94bcad4b0fbc8f3bac306cd1c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 150], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 256, 150], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_59040adfe86ffe21a929d9ce9a414836(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 24, 36]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 24, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_21108bb32226466f264753fdeda43fc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59040adfe86ffe21a929d9ce9a414836
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 24, 36], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_21108bb32226466f264753fdeda43fc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59040adfe86ffe21a929d9ce9a414836
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 24, 36], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_d6a0d913dbbc59fc643658bd710a121a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33aacc84707d04092c21218cf349a602
        def get_inputs(self):
            return [
                paddle.uniform([1, 150, 384], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_72a8f7c0cf80e6dbbe0081fd45d74547(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 256, 21]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 21], dtype='float32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa66ed6f90fe2c9244cbe01d88c4514f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72a8f7c0cf80e6dbbe0081fd45d74547
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 256, 21], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_1e38e8ad2a2febb28ad77b1985a22624(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 25, 38]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 25, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_02c237870cbe422469cade972ae6dd73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e38e8ad2a2febb28ad77b1985a22624
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 25, 38], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_02c237870cbe422469cade972ae6dd73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e38e8ad2a2febb28ad77b1985a22624
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 25, 38], dtype='int64').reshape([4]),
            ]


    
    class PrimitiveOp_0953d0070ba10f4f49dfe4ecb94c8458(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 20, 30]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 20, 30], dtype='float32'),
                paddle.static.InputSpec(shape=[4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_82745e61ee27cb3f61b43c3947b6e981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0953d0070ba10f4f49dfe4ecb94c8458
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 20, 30], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_82745e61ee27cb3f61b43c3947b6e981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0953d0070ba10f4f49dfe4ecb94c8458
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 20, 30], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_9126bd6a1495c02150b89aaed372cec2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5dd9d14cc26b0a8447bf0818b5cef84
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 768], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_39a0a5499c5ef1f42abbe53afc932c65(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [7, 256, 19]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_83cc400a3bc033c2a5ae2558a0e517d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39a0a5499c5ef1f42abbe53afc932c65
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 256, 19], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_826abbf55a7ce8c7379f92ca3f02b303(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 15, 25]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 15, 25], dtype='float32'),
                paddle.static.InputSpec(shape=[4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f67a3681ba4adb19f769584451f3e68b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_826abbf55a7ce8c7379f92ca3f02b303
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 15, 25], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_f67a3681ba4adb19f769584451f3e68b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_826abbf55a7ce8c7379f92ca3f02b303
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 15, 25], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_5d3b5e692c03b8a0af7231db97038d6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84e352f55ad21d125e47ade527c7a776
        def get_inputs(self):
            return [
                paddle.uniform([1, 150, 768], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_88aa5597a6e747cf543a5ec5baba23fa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [8, 256, 150]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 150], dtype='float32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f62a4cbf9a21634a6c9ac166bf37f3ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88aa5597a6e747cf543a5ec5baba23fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 150], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 256, 150], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_35b3da4bc39e8bde22fb3860026a02b7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 24, 36]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eda006635e168fa80818ba9e500be0a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35b3da4bc39e8bde22fb3860026a02b7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 24, 36], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_eda006635e168fa80818ba9e500be0a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35b3da4bc39e8bde22fb3860026a02b7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 24, 36], dtype='int64').reshape([4]),
            ]


    
    class PrimitiveOp_dc977d7f2dd3fe9bfdcebd144e51df0d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, -1, -1]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bea8550066a0c616f7e60321c16c9e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc977d7f2dd3fe9bfdcebd144e51df0d
        def get_inputs(self):
            return [
                paddle.uniform([1, 150, 384], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_cdb4a0ec7099edf78a1d712efb3f1da7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 256, 21]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cc1a5a8f993af8bebdecb8dff2285639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdb4a0ec7099edf78a1d712efb3f1da7
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 256, 21], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_581dd08af461e9d30cb35060ed17232c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 25, 38]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_07c3c6b7fb0f2fa71338b46f71edda20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_581dd08af461e9d30cb35060ed17232c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 25, 38], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_07c3c6b7fb0f2fa71338b46f71edda20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_581dd08af461e9d30cb35060ed17232c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 25, 38], dtype='int64').reshape([4]),
            ]


    
    class PrimitiveOp_8c721723d52eac43bc9942b7f8ae0981(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 20, 30]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44d2306e6156ec1457808e4926b08aef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c721723d52eac43bc9942b7f8ae0981
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 20, 30], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_44d2306e6156ec1457808e4926b08aef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c721723d52eac43bc9942b7f8ae0981
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 20, 30], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_28a1d1c315bbb9888da711ec6ab67b08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc977d7f2dd3fe9bfdcebd144e51df0d
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 768], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_c53c3232f2ebb7608823f177d1da59da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [7, 256, 19]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_95b9689c79dc3c3f6b891f2e692f1503(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c53c3232f2ebb7608823f177d1da59da
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 256, 19], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_c7acd36e217da4b620d2655e238364d4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 15, 25]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_df5e6ce2827f7e003eb4d0362f264bff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7acd36e217da4b620d2655e238364d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 15, 25], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_df5e6ce2827f7e003eb4d0362f264bff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7acd36e217da4b620d2655e238364d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 15, 25], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_3619c716a5eb4d06f1d11fa02649e35c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc977d7f2dd3fe9bfdcebd144e51df0d
        def get_inputs(self):
            return [
                paddle.uniform([1, 150, 768], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_5c13b5f2aae12abdf8299f598ab6b010(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [8, 256, 150]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a7f82e86f8e8d4c30fb458f7f274698(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c13b5f2aae12abdf8299f598ab6b010
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 150], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 256, 150], dtype='int64').reshape([3]),
            ]


    

if __name__ == '__main__':
    unittest.main()