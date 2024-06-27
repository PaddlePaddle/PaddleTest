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
    class PrimitiveOp_681c77f74a72b7979cdf63f97749694f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 7, 7, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e5135d7b76b3b1a79e64aaf1856f6e29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_681c77f74a72b7979cdf63f97749694f
        def get_inputs(self):
            return [
                paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c1a9e8d2d34a9b24426a12428212fce2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 14, 14, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7013e73c9626b818e306b98c8f2dcaa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1a9e8d2d34a9b24426a12428212fce2
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e0647b7086d49fcc543b69c64477e866(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_681c77f74a72b7979cdf63f97749694f
        def get_inputs(self):
            return [
                paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_0c35e30dc994c67bce7b1fa10bb04301(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 56, 56, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ee74a408cc66a56ce9bed27d5dcb5cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c35e30dc994c67bce7b1fa10bb04301
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_f7fd2101960f7226a458a9c510687fe5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 28, 28, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_606535ff178eab9c714fec94ba7cebf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fd2101960f7226a458a9c510687fe5
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7013e73c9626b818e306b98c8f2dcaa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1a9e8d2d34a9b24426a12428212fce2
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7ed8b443637232f96c40fd13e182f063(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1a9e8d2d34a9b24426a12428212fce2
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d76550971ecd9fb47095271416667b18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c35e30dc994c67bce7b1fa10bb04301
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_566efae57c5accf7255934cf751b7d89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fd2101960f7226a458a9c510687fe5
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7ed8b443637232f96c40fd13e182f063(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1a9e8d2d34a9b24426a12428212fce2
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_164236db67858ac7ea7043dbc866c90f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 7, 7, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc7a2a707e0753baae397734f592ce5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_164236db67858ac7ea7043dbc866c90f
        def get_inputs(self):
            return [
                paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_16aa823b3d58ffdc3817abdbd70b47d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 14, 14, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_37571b4ae374f98c23a9577a52c90100(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16aa823b3d58ffdc3817abdbd70b47d6
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e85535253ca460fb679dfd759a04b1f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 7, 7, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_975c852d01a435d91ae0fdc9591156a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85535253ca460fb679dfd759a04b1f5
        def get_inputs(self):
            return [
                paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_eba4700132dff33b887a21f0fe81a626(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 56, 56, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5297a002a886d43d0d70f4eb75ddc970(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eba4700132dff33b887a21f0fe81a626
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_22e38993959b1e642d1b8bfc621445a8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 28, 28, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9389e5d3217b1cfe123bdbe0ce2d44b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22e38993959b1e642d1b8bfc621445a8
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_37571b4ae374f98c23a9577a52c90100(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16aa823b3d58ffdc3817abdbd70b47d6
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_87768f3211083ff225a1046a161d8479(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 14, 14, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_27ae0b14b799780404fda12d3836b331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87768f3211083ff225a1046a161d8479
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_aa00b2378f78ec0b576ed73f4634de6a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 56, 56, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_223c2a26ca3a1874d32022abc8da22bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa00b2378f78ec0b576ed73f4634de6a
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bb1b94897522b21f122a4434cf8b2789(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 28, 28, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_db65c3e376aaa623f55b03fe9e3b3eca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb1b94897522b21f122a4434cf8b2789
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_27ae0b14b799780404fda12d3836b331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87768f3211083ff225a1046a161d8479
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a476673266540b2ce16f898fd25e4bb6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd5a9cc4b025b6a0bfee51eac44b8add(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a476673266540b2ce16f898fd25e4bb6
        def get_inputs(self):
            return [
                paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_66e165945caae00d3665e0d27a499420(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a476673266540b2ce16f898fd25e4bb6
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7aac89da1eb6b0bdc2eb27169814214b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a476673266540b2ce16f898fd25e4bb6
        def get_inputs(self):
            return [
                paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f69232acc63c305f62028aa86d5ed5b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a476673266540b2ce16f898fd25e4bb6
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_73f7f72ca9ac6a4019c9d213698b7bd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a476673266540b2ce16f898fd25e4bb6
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_66e165945caae00d3665e0d27a499420(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a476673266540b2ce16f898fd25e4bb6
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_dc4d9d915dee1748a35bc1e248be7c5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a476673266540b2ce16f898fd25e4bb6
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5a6b99a39b2c5f8a02f8074525fc6200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a476673266540b2ce16f898fd25e4bb6
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_94d6f53e5800c4b16d509c56f4b59de5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a476673266540b2ce16f898fd25e4bb6
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_dc4d9d915dee1748a35bc1e248be7c5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a476673266540b2ce16f898fd25e4bb6
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
            ]


    

if __name__ == '__main__':
    unittest.main()