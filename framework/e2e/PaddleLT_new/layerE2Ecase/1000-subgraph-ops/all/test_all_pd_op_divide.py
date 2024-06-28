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
    class PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f11be99b9ddb9b0670ad0cfa921480fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_398c814fc0b8c3ce8cb065fc6d81a4d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_adf42b04d28c3f6e2d920614a9ca9273(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4fa4900ce8f49e85eeb359285595beb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adf42b04d28c3f6e2d920614a9ca9273
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.23544012010097504]]], dtype='float32').reshape([1, 1, 1]),
            ]


    
    class PrimitiveOp_2688bff273bba9d859b559dfa0e917ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[12096, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[12096, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6eb83423e258b588af18ea1622d86d8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2688bff273bba9d859b559dfa0e917ee
        def get_inputs(self):
            return [
                paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([12096, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a27c71a872ee36b9db66693cadc7729(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af94317360d7b61d66ac1d8a04bffc2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_151713f40442abd0afb75fc9e3ad549a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6b6fd98b82b4150b1d44f7ed45e6ce47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_151713f40442abd0afb75fc9e3ad549a
        def get_inputs(self):
            return [
                paddle.to_tensor([1076.5631103515625], dtype='float32').reshape([1]),
                paddle.to_tensor(8732.0, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c352c426ff9235df85ccf44de562513c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.00040858826832845807], [2.772476136669866e-06], [7.675283995922655e-05], [0.001655016327276826], [0.0001419818727299571], [0.018362853676080704]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_4fa3bf37f93cbba8f4684cbcb27e009e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[1.3540058716898784e-05], [1.6532496374566108e-05], [0.005603802856057882], [0.0017022115644067526], [0.010291928425431252], [0.0005788452690467238]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_76e3312e6875dde0ea52b282757a3247(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adf42b04d28c3f6e2d920614a9ca9273
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.07576718181371689], [0.00447469437494874], [0.0977160707116127], [0.06420623511075974], [0.10864578187465668], [0.11122038960456848]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_8c3aa0947d47cd59e0d977811c557d9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_247550de1e8857b28f6a7e7f8e51b226(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f11be99b9ddb9b0670ad0cfa921480fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d77be0af403dcf7788ce7056ba84d11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_23138ba4175e40da076559a9914ccb4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(8.992445945739746, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f5532aa571f5880479b5bbd8b77c10cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(2.4994864463806152, dtype='float32').reshape([]),
                paddle.to_tensor([2.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_af2f7c71c3e55f9b2398a3b6b365f747(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cb83cf4564525eb540f14e838b36b197(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0820d5b49481455044f27d23d57c8377(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0820d5b49481455044f27d23d57c8377(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9be790f68a6a864a2768cd7745f90f69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(183327.234375, dtype='float32').reshape([]),
                paddle.to_tensor([0.42846009135246277], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_00a90afbb38e95d112fe346d8a10f955(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(97768.9921875, dtype='float32').reshape([]),
                paddle.to_tensor([0.42846009135246277], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4ba6e0ca6f3d4e7186af66db88a64437(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(940.4869384765625, dtype='float32').reshape([]),
                paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_95e715cd595f6f77cd7d5a58ecd6da29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c57f726ad15091268e7f32626ce4fb56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b3d40918a11d29fcc86da0051f10e7ba(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5376, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[5376, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c1a1dde4d94209141982da4ee783a48a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3d40918a11d29fcc86da0051f10e7ba
        def get_inputs(self):
            return [
                paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([5376, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad141b373c8b910e34512356a42e8677(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d889ca03446a56505675d2ca9c34c82d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_32a9f90c8f1315312bff584b2a0bf8ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.010175604373216629], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.05231598764657974], [0.009299254976212978], [0.10837691277265549], [-0.07017733156681061], [0.07759939134120941], [-0.016110287979245186], [-0.0599064826965332], [0.08122386783361435], [-0.026601403951644897]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_5ca1050423aa4848e7e940950abb0419(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0013923496007919312], [0.029073655605316162], [-0.003807641565799713], [0.1609496772289276], [0.020773865282535553], [0.02696826308965683], [0.045601993799209595], [-0.08601643890142441], [0.10990434885025024]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.05092363804578781], [0.038372911512851715], [0.10456927120685577], [0.090772345662117], [0.09837325662374496], [0.010857976041734219], [-0.014304488897323608], [-0.004792569670826197], [0.08330294489860535]], dtype='float32').reshape([9, 1]),
            ]


    
    class PrimitiveOp_f73f8a5a9cc5d1ba89df5ec87c9b6c49(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_959e90bb7b91c728229da3680a993efe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f73f8a5a9cc5d1ba89df5ec87c9b6c49
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.12166859954595566], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_56a6aa7bbf8120709a041d986a349061(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56a6aa7bbf8120709a041d986a349061(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c12bfa685ddf9243c72af4613d70fc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(32006.1796875, dtype='float32').reshape([]),
                paddle.to_tensor([0.10051140189170837], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_410357a450ec4fa17cd151a7acc1f4b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(3933.03173828125, dtype='float32').reshape([]),
                paddle.to_tensor([0.10051140189170837], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_b1c61b24e5e854c926c41c52f551490d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c594b7c5814e628652f0dd897c87ea8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, -0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
                paddle.to_tensor([0.020324425771832466, 0.013810954988002777, -0.004959273152053356, 0.040052562952041626, -0.04735404625535011, -0.022230884060263634], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_5f88f079b94c237fd71c59c171349645(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.014093050733208656, 0.05383152514696121, 0.031237879768013954, 0.005727421957999468, 0.03170868754386902, 0.040966905653476715], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0018855603411793709, 0.30608075857162476, 0.14130333065986633, 0.08980018645524979, 0.04460727050900459, 0.04448986425995827], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_2b8aa923331b21d5d594dd2f6f2e70e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.16486512124538422, 0.04520697519183159, -0.02553534507751465, -0.0034683942794799805, 0.11817975342273712, -0.3263065814971924], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.12327910959720612, 0.3055049479007721, 0.19421210885047913, -0.022199705243110657, -0.4006950855255127, 0.06812882423400879], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e4beee466ee4252bbda262c4765a0582(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.03921353816986084, 0.3743213713169098, 0.1907661110162735, 0.2556256353855133, 0.05274197459220886, -0.13855475187301636], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0158902108669281, -0.21657240390777588, -0.13378392159938812, 0.15638324618339539, -0.1620321273803711, 0.026508353650569916], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8e5a2c4b8af2638ce6f1a3e11bb5d9ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([1.812121868133545, 0.5770043730735779, 0.27817100286483765, 0.3044983446598053, 0.0003150637785438448, 0.00011431847815401852], dtype='float32').reshape([6]),
                paddle.to_tensor([2.812121868133545, 1.5770044326782227, 1.2781710624694824, 1.304498314857483, 1.0003150701522827, 1.0001143217086792], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_3590c3a988c28bbb961b929b3fe88a4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3590c3a988c28bbb961b929b3fe88a4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b39865e92dc5c8ee48f3856609ef9b0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-1822.7001953125, dtype='float32').reshape([]),
                paddle.to_tensor([0.27057820558547974], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4e0c202fe799d1ca4173ab72375666c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(103763.359375, dtype='float32').reshape([]),
                paddle.to_tensor([0.27057820558547974], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_174b027b5bea4a74e1ba999470d656d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(958.8797607421875, dtype='float32').reshape([]),
                paddle.to_tensor([4.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_9c749ed8d8687331033eba5eb81320ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8400, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[8400, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_97d561d781434abef1c8358ad896eac4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c749ed8d8687331033eba5eb81320ec
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([8400, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ab330027ee849c5d0679b8ab1722ad4e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1012f6e6a4fc8cd38f4b82161d356bd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab330027ee849c5d0679b8ab1722ad4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af94317360d7b61d66ac1d8a04bffc2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c57f726ad15091268e7f32626ce4fb56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c8f68c3235ad43f7311c65518d1f049(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c8f68c3235ad43f7311c65518d1f049(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d75af67acb11d9d294925910e1bb408(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-711917.6875, dtype='float32').reshape([]),
                paddle.to_tensor([0.21900232136249542], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6851299f29c9e7719330c974f9602a58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(87456.15625, dtype='float32').reshape([]),
                paddle.to_tensor([0.21900232136249542], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5f84836bfe68de38f2ee12e0441d8c7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adf42b04d28c3f6e2d920614a9ca9273
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.23765969276428223], [0.24743786454200745]]], dtype='float32').reshape([1, 2, 1]),
            ]


    class TestPrimitiveOp_247550de1e8857b28f6a7e7f8e51b226(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_398c814fc0b8c3ce8cb065fc6d81a4d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4da99e9a95721386fa7510d5e95d9d1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0011826036497950554]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_853d3085cebab6c1b16612821379758f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03306867927312851]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.03425128385424614]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_8f5b940a731d2ab5704202cbd17461e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.014378788881003857], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.24575075507164001], [-0.04205697402358055], [0.0034978860057890415], [0.003514286130666733], [0.1535715013742447], [0.1216726154088974]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_8ca0f8fc631031e2f36eef75ef177a57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.15333926677703857], [0.06525279581546783], [-0.00631610956043005], [0.02273622713983059], [0.023386597633361816], [-0.10779407620429993]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.09241148084402084], [0.023195823654532433], [-0.002818223787471652], [0.026250513270497322], [0.1769580990076065], [0.013878539204597473]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_a3dafceb9fa2322512874ea7c9c4595e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adf42b04d28c3f6e2d920614a9ca9273
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24484822154045105]]], dtype='float32').reshape([1, 1, 1]),
            ]


    
    class PrimitiveOp_bf6bebb53e4bc6915f8ba960dcf7ea77(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7658c76c456395c605011037d5aef3ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf6bebb53e4bc6915f8ba960dcf7ea77
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95e715cd595f6f77cd7d5a58ecd6da29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a27c71a872ee36b9db66693cadc7729(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5432239bb8fb11b62e5977411ea1238c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(60.11579132080078, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_374bc874a4f085ff522ad9097525b045(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(545.63427734375, dtype='float32').reshape([]),
                paddle.to_tensor([4.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_587718919ee1ababc5f81fb9971d6f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587718919ee1ababc5f81fb9971d6f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_420ac1909858b9c4ef3fcd8cdbb6ad76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-67143.1796875, dtype='float32').reshape([]),
                paddle.to_tensor([0.322158545255661], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_22a9ae4ff1378e39353200acd270217d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(121165.015625, dtype='float32').reshape([]),
                paddle.to_tensor([0.322158545255661], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_132fa05f875a6614ff7c5cc0a3fb5817(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf6bebb53e4bc6915f8ba960dcf7ea77
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f17c4071a87bab74c185606e898b17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80608115b9a62e552b71da76d148f17d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f73f8a5a9cc5d1ba89df5ec87c9b6c49
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3232676386833191], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_277900458982b42a56ac1c07ee6165b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_277900458982b42a56ac1c07ee6165b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b380b1207d3bff2c4fbaaa8dc75d4cba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-729765.5, dtype='float32').reshape([]),
                paddle.to_tensor([0.204777792096138], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a2196964ea038e27a83ad97478f64f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(261511.6875, dtype='float32').reshape([]),
                paddle.to_tensor([0.204777792096138], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_751b315a0a607ef1137a72cc3b9d7e0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_151713f40442abd0afb75fc9e3ad549a
        def get_inputs(self):
            return [
                paddle.to_tensor([305.69525146484375], dtype='float32').reshape([1]),
                paddle.to_tensor(2434.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_8bf9ff5ef492d2ff54097e0d1b23d63a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bf9ff5ef492d2ff54097e0d1b23d63a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8be04bdc089ec0d4a331d2c7d81de553(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-19123.427734375, dtype='float32').reshape([]),
                paddle.to_tensor([0.3438152074813843], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_65ab386387cc112502819159e85cfd82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(14883.05078125, dtype='float32').reshape([]),
                paddle.to_tensor([0.3438152074813843], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8c3aa0947d47cd59e0d977811c557d9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_28f13cea8570ff79d597bb01974d28ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[100, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1fd2e10826294b914e96c0b479f82ca2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28f13cea8570ff79d597bb01974d28ef
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f12ad5dcaea7d41bca1b5e27df5083e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0020067794248461723, 0.2930905520915985, 0.2942022681236267, 0.4206882417201996], [0.022792115807533264, 0.4363101124763489, 0.3274344801902771, 0.033717453479766846]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.033379796892404556, 0.010689612478017807, 0.1018703281879425, 0.29704639315605164], [0.1340337097644806, 0.3128531277179718, 0.4129876494407654, 0.4426139295101166]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_ad141b373c8b910e34512356a42e8677(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_370b7b27371cf0eb63020a3f31061bd4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6069, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[6069, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ddecfc268c66168c9d6bc0cd786e0e57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_370b7b27371cf0eb63020a3f31061bd4
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([6069, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c03566a731d3e775dd3575bbd5e8678f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[300, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_36cf1bfd297093df744ffed6795767d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c03566a731d3e775dd3575bbd5e8678f
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08a7636bc75ae2b6a71b588ab5d176f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3937280774116516, 0.4412074089050293, 0.34095263481140137, 0.47780323028564453], [0.2755979895591736, 0.22411662340164185, 0.48828834295272827, 0.0838293507695198]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.34371617436408997, 0.4690535068511963, 0.36261194944381714, 0.16057159006595612], [0.02885439246892929, 0.22169719636440277, 0.21083924174308777, 0.35290396213531494]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_816e8ae754700e7dc642d15724a8ed8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.005169573239982128], [0.06563594937324524], [0.019044138491153717], [0.1293209046125412], [-0.01654825359582901]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_1dbf3a943bf1a37f33c707836bae69be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.017894238233566284], [0.021296389400959015], [0.09752587974071503], [-0.036712780594825745], [0.0270785354077816]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.012724664062261581], [0.08693233877420425], [0.11657001823186874], [0.09260812401771545], [0.010530282743275166]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_dcfe93db9305b638b8f915254642ab58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f73f8a5a9cc5d1ba89df5ec87c9b6c49
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3920208513736725], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_af2f7c71c3e55f9b2398a3b6b365f747(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d889ca03446a56505675d2ca9c34c82d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_def5bbb48d4660d6672b91763f078ce8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3913da6994244f1a5a49f3c0e9f4bea7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3913da6994244f1a5a49f3c0e9f4bea7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83c93653f576564e74d3a8914b453660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-11323224.0, dtype='float32').reshape([]),
                paddle.to_tensor([0.32941845059394836], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9423e92934283d239c8b04adf5284f0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(133113.625, dtype='float32').reshape([]),
                paddle.to_tensor([0.32941845059394836], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f7ffef4df666cf498a6cd152ed6e482a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7ffef4df666cf498a6cd152ed6e482a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81df1d32c0c915a6d5ef971579b49a7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(193308.859375, dtype='float32').reshape([]),
                paddle.to_tensor([0.11696609854698181], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_61ec547677c6ca3371444a9921e02ebc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(174317.390625, dtype='float32').reshape([]),
                paddle.to_tensor([0.11696609854698181], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d5ff631595f29a045a4c83a9884b7038(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5ff631595f29a045a4c83a9884b7038(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d82639ad8a1ee7dc1644f7d09344c13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(604421.6875, dtype='float32').reshape([]),
                paddle.to_tensor([0.19147957861423492], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8ecdb9f4c3ecb5b4ee01c0f386b44d68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(217196.53125, dtype='float32').reshape([]),
                paddle.to_tensor([0.19147957861423492], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_810671423ad99cb03c30bf72989a9950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f73f8a5a9cc5d1ba89df5ec87c9b6c49
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.21355828642845154], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9f17c4071a87bab74c185606e898b17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b336b4dbaf31bffcc1b90cafac29ef08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(14.848526000976562, dtype='float32').reshape([]),
                paddle.to_tensor([3.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_9c4f0ba0d4034fceae84d75338b628ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20267, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[20267, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dea8dd511af834402484677a1d04f743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c4f0ba0d4034fceae84d75338b628ae
        def get_inputs(self):
            return [
                paddle.uniform([20267, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([20267, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75fb3d314147a39880fcd7dc8c97ae54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.02299167960882187], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-0.04221170023083687], [0.14200380444526672], [-0.0011601647129282355], [-0.019290968775749207]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_94e0339b217dbecd4be18300f5cc1b9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.053318336606025696], [0.01920013129711151], [0.008827432990074158], [0.08699016273021698]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.011106634512543678], [0.16120393574237823], [0.0076672681607306], [0.06769919395446777]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f9454ebe1ac2c89c649ed852708923f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(4.3850321769714355, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f6e33f34b67bb24be24e34895139aa8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6e33f34b67bb24be24e34895139aa8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27eb7713b7fba53189a2151a5e6e9fa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(251715.625, dtype='float32').reshape([]),
                paddle.to_tensor([0.1501390039920807], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5ff604726845472e068cc9eef187c073(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(29097.33984375, dtype='float32').reshape([]),
                paddle.to_tensor([0.1501390039920807], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_32f6c46e4e861158e8213839d7a1a26b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f73f8a5a9cc5d1ba89df5ec87c9b6c49
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3421512842178345], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4cba35131fbdf5210ee22eb97dc8db05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(33.76018142700195, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_faf92cc7dea26277cdc049216c2d32ed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6804, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[6804, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c181afd1cd123a0e1b640ddb664e67e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_faf92cc7dea26277cdc049216c2d32ed
        def get_inputs(self):
            return [
                paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([6804, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_def5bbb48d4660d6672b91763f078ce8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d70de17803062049f8b6b540681f4bb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(237.84315490722656, dtype='float32').reshape([]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_836dd311c2f8230733a2d5269209b2e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(138.18959045410156, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2d77be0af403dcf7788ce7056ba84d11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79d79066cdddc014bcdadbf29f624cf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c86575c4f044cafef805140374a4987(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c86575c4f044cafef805140374a4987(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62ed930f8b205d567cec9c078c939206(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(47518.54296875, dtype='float32').reshape([]),
                paddle.to_tensor([0.3970640003681183], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_df13305f12f01d8b524aa5a05ba2ae9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(238131.71875, dtype='float32').reshape([]),
                paddle.to_tensor([0.3970640003681183], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_79d79066cdddc014bcdadbf29f624cf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cc3b3360da675eee1cd90fcdc82f3540(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bcbc8f72b55de675e5635b786681ed61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3b3360da675eee1cd90fcdc82f3540
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f7ceb231e6ea0fc656ca99baddbe27d4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd985d01f03d958f04bedd107a1741f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7ceb231e6ea0fc656ca99baddbe27d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c905953f726f19f90c41ef4927424cef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 2100], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a79bf514a31bbf0d6574e71f2696d444(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c905953f726f19f90c41ef4927424cef
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.23544012010097504]]], dtype='float32').reshape([1, 1, 1]),
            ]


    class TestPrimitiveOp_6eb83423e258b588af18ea1622d86d8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2688bff273bba9d859b559dfa0e917ee
        def get_inputs(self):
            return [
                paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([12096, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c107832b4157be7086f9ee5ecf934921(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d6e5e3d1d4343fe04d9a362d43301dee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c107832b4157be7086f9ee5ecf934921
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_85ab518f0357cf16c4430df9063f61ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6921bd8c69cd727a75e33168f1f2c8e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85ab518f0357cf16c4430df9063f61ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6cdac9f6ef2c4003ed867abb8507f929(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f964f35bab717097c4206fbffb502cd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cdac9f6ef2c4003ed867abb8507f929
        def get_inputs(self):
            return [
                paddle.to_tensor([1076.5631103515625], dtype='float32').reshape([1]),
                paddle.to_tensor(8732.0, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_ca3dbde71d4cbf7e3bd6d9616fec7e12(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 21824], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 6, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d6d8aa808d681b881ace865379515954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca3dbde71d4cbf7e3bd6d9616fec7e12
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.00040858826832845807], [2.772476136669866e-06], [7.675283995922655e-05], [0.001655016327276826], [0.0001419818727299571], [0.018362853676080704]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_deb8aa1d1f213e0cdcd3a950c4defe6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca3dbde71d4cbf7e3bd6d9616fec7e12
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[1.3540058716898784e-05], [1.6532496374566108e-05], [0.005603802856057882], [0.0017022115644067526], [0.010291928425431252], [0.0005788452690467238]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_def4cd188e760ecfcff6c5649c2d2a31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca3dbde71d4cbf7e3bd6d9616fec7e12
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.07576718181371689], [0.00447469437494874], [0.0977160707116127], [0.06420623511075974], [0.10864578187465668], [0.11122038960456848]]], dtype='float32').reshape([1, 6, 1]),
            ]


    
    class PrimitiveOp_b755566a0ec2e9481405ddcb54faa9da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_61b7dfbd95a5b9e983a211e4e7ed9dd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b755566a0ec2e9481405ddcb54faa9da
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0273f2882c489a087c837a42baf28563(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_57ee7d00faab573dd52c8a263e6011ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0273f2882c489a087c837a42baf28563
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcbc8f72b55de675e5635b786681ed61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3b3360da675eee1cd90fcdc82f3540
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_98833d7b4c14a0e4c867033e684c3a13(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff40957fd977a6a0d2571b5ff7ba8a44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98833d7b4c14a0e4c867033e684c3a13
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fb461ed1473e708df86d4965d867f485(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6d3b56d86794a7299940eade6f07828(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(8.992445945739746, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a8e8e86b924293faf43091f036e5f1b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(2.4994864463806152, dtype='float32').reshape([]),
                paddle.to_tensor([2.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_636b2116e305caaee871cb19c45ea54a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f2ae46f17671088fd9043666fea3299(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_636b2116e305caaee871cb19c45ea54a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cd7fc68eae6c80f201448a6451f8a7a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1723, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1723, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_27381b84defc521d023c6b3280b70fe9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd7fc68eae6c80f201448a6451f8a7a3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27381b84defc521d023c6b3280b70fe9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd7fc68eae6c80f201448a6451f8a7a3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48c56fcb101360767ae9028b147d0d86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(183327.234375, dtype='float32').reshape([]),
                paddle.to_tensor([0.42846009135246277], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c3c316b194facb2942e0462488784de4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(97768.9921875, dtype='float32').reshape([]),
                paddle.to_tensor([0.42846009135246277], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b77a8d8f178d367cee6c154d8a4a8f9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(940.4869384765625, dtype='float32').reshape([]),
                paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_7385aec8a9e86495eec841db72d78256(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0782fd68b15ee604fe265479d49807b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7385aec8a9e86495eec841db72d78256
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f0662aceec27ef3ef08ee66dcd56d8b3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_00ec08dbcea99e39920a89480f60d2a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0662aceec27ef3ef08ee66dcd56d8b3
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1a1dde4d94209141982da4ee783a48a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3d40918a11d29fcc86da0051f10e7ba
        def get_inputs(self):
            return [
                paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([5376, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ae683b13bea09acdea6c3e1c0be2eab0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e4d5e4dbeb753f9be751f37fe8979e6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae683b13bea09acdea6c3e1c0be2eab0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_376de55309d118e1fd82e62f7f01348c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e6ef9f609328c6d1440af655d041f52e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_376de55309d118e1fd82e62f7f01348c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_68ded890f87b9d3fd6c362b4fcfc0e1b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_964437c37a3a0ec080e312e8219449f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68ded890f87b9d3fd6c362b4fcfc0e1b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.010175604373216629], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.05231598764657974], [0.009299254976212978], [0.10837691277265549], [-0.07017733156681061], [0.07759939134120941], [-0.016110287979245186], [-0.0599064826965332], [0.08122386783361435], [-0.026601403951644897]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_4da15a02c65e046477b4731971355bad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68ded890f87b9d3fd6c362b4fcfc0e1b
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0013923496007919312], [0.029073655605316162], [-0.003807641565799713], [0.1609496772289276], [0.020773865282535553], [0.02696826308965683], [0.045601993799209595], [-0.08601643890142441], [0.10990434885025024]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.05092363804578781], [0.038372911512851715], [0.10456927120685577], [0.090772345662117], [0.09837325662374496], [0.010857976041734219], [-0.014304488897323608], [-0.004792569670826197], [0.08330294489860535]], dtype='float32').reshape([9, 1]),
            ]


    
    class PrimitiveOp_fea08d98e460778978a09aef3cd730f6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 8, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d09d4ddab6684ffee48821d45864805a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fea08d98e460778978a09aef3cd730f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.12166859954595566], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_bdbfa0af2daf95fb8352b3581dc699a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5498, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[5498, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_991f9316778688e7c37e9aea0cf6395e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bdbfa0af2daf95fb8352b3581dc699a9
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_991f9316778688e7c37e9aea0cf6395e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bdbfa0af2daf95fb8352b3581dc699a9
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a2622c377625c42e0054c6a0cb5e0a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(32006.1796875, dtype='float32').reshape([]),
                paddle.to_tensor([0.10051140189170837], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bd5a3f1a0f46412957deff6cf5ba9e55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(3933.03173828125, dtype='float32').reshape([]),
                paddle.to_tensor([0.10051140189170837], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_991840637b016f1ceba5f8049ac0c8ed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6], dtype='float32'),
                paddle.static.InputSpec(shape=[6], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e5239799de73b7037e11c73d9b2f995e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_991840637b016f1ceba5f8049ac0c8ed
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, -0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
                paddle.to_tensor([0.020324425771832466, 0.013810954988002777, -0.004959273152053356, 0.040052562952041626, -0.04735404625535011, -0.022230884060263634], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_b030673aec7fa5c952f8223dcc4f7b52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_991840637b016f1ceba5f8049ac0c8ed
        def get_inputs(self):
            return [
                paddle.to_tensor([0.014093050733208656, 0.05383152514696121, 0.031237879768013954, 0.005727421957999468, 0.03170868754386902, 0.040966905653476715], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0018855603411793709, 0.30608075857162476, 0.14130333065986633, 0.08980018645524979, 0.04460727050900459, 0.04448986425995827], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_7bab9bf7f8bf47f036b13bb0bb5a245d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_991840637b016f1ceba5f8049ac0c8ed
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.16486512124538422, 0.04520697519183159, -0.02553534507751465, -0.0034683942794799805, 0.11817975342273712, -0.3263065814971924], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.12327910959720612, 0.3055049479007721, 0.19421210885047913, -0.022199705243110657, -0.4006950855255127, 0.06812882423400879], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_5b40856c53d4b203a68bdbcbddfb60d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_991840637b016f1ceba5f8049ac0c8ed
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.03921353816986084, 0.3743213713169098, 0.1907661110162735, 0.2556256353855133, 0.05274197459220886, -0.13855475187301636], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0158902108669281, -0.21657240390777588, -0.13378392159938812, 0.15638324618339539, -0.1620321273803711, 0.026508353650569916], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_6c29d07b7c0a9fa2bb5adbea8bca6573(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_991840637b016f1ceba5f8049ac0c8ed
        def get_inputs(self):
            return [
                paddle.to_tensor([1.812121868133545, 0.5770043730735779, 0.27817100286483765, 0.3044983446598053, 0.0003150637785438448, 0.00011431847815401852], dtype='float32').reshape([6]),
                paddle.to_tensor([2.812121868133545, 1.5770044326782227, 1.2781710624694824, 1.304498314857483, 1.0003150701522827, 1.0001143217086792], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_a2e3b72bd8b820ecfeee24e943141484(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1759, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1759, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f373353159d4255c593e13d698247505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a2e3b72bd8b820ecfeee24e943141484
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f373353159d4255c593e13d698247505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a2e3b72bd8b820ecfeee24e943141484
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3de68067b2a991fa99974a485a7e44a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(-1822.7001953125, dtype='float32').reshape([]),
                paddle.to_tensor([0.27057820558547974], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8f63005f740e2bddb07136f39b80e7cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(103763.359375, dtype='float32').reshape([]),
                paddle.to_tensor([0.27057820558547974], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_73d5457538964421dd7ffb83b104b252(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(958.8797607421875, dtype='float32').reshape([]),
                paddle.to_tensor([4.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_97d561d781434abef1c8358ad896eac4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c749ed8d8687331033eba5eb81320ec
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([8400, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a27175182f06d2332fc7bbd881f58d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 38, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 38, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e3c531d67bbe4286af06e822676a84cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a27175182f06d2332fc7bbd881f58d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6921bd8c69cd727a75e33168f1f2c8e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85ab518f0357cf16c4430df9063f61ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00ec08dbcea99e39920a89480f60d2a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0662aceec27ef3ef08ee66dcd56d8b3
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ba80d9444ea946856f8eaa19bb2e5fde(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1538, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1538, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d78d4eb463a7eccbd706efca83840e5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba80d9444ea946856f8eaa19bb2e5fde
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d78d4eb463a7eccbd706efca83840e5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba80d9444ea946856f8eaa19bb2e5fde
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30094ca6f8479f19ffa411cf217275e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(-711917.6875, dtype='float32').reshape([]),
                paddle.to_tensor([0.21900232136249542], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d16d8ed56622f31a69f356139c251cad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(87456.15625, dtype='float32').reshape([]),
                paddle.to_tensor([0.21900232136249542], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_9d2e452b8e4207992fdbf0ad3db96a44(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 3549], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_10a07748e384554145cbf10cdc27c6f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d2e452b8e4207992fdbf0ad3db96a44
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.23765969276428223], [0.24743786454200745]]], dtype='float32').reshape([1, 2, 1]),
            ]


    class TestPrimitiveOp_57ee7d00faab573dd52c8a263e6011ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0273f2882c489a087c837a42baf28563
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd985d01f03d958f04bedd107a1741f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7ceb231e6ea0fc656ca99baddbe27d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e074e766a0ef5ac615b8de21d9514981(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_323a1ddf2549800056b98a83a35db338(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e074e766a0ef5ac615b8de21d9514981
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0011826036497950554]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_7ea51a0a0e5835bbe4f14bf11fb9483d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e074e766a0ef5ac615b8de21d9514981
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03306867927312851]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.03425128385424614]], dtype='float32').reshape([1, 1]),
            ]


    
    class PrimitiveOp_8d618a1b4a2ab12bf286cb1ad270ff0e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c5435b963b02fb4de833abc845c478ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d618a1b4a2ab12bf286cb1ad270ff0e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.014378788881003857], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.24575075507164001], [-0.04205697402358055], [0.0034978860057890415], [0.003514286130666733], [0.1535715013742447], [0.1216726154088974]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_f2b36299764de74c0c6ee80b5c11ad45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d618a1b4a2ab12bf286cb1ad270ff0e
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.15333926677703857], [0.06525279581546783], [-0.00631610956043005], [0.02273622713983059], [0.023386597633361816], [-0.10779407620429993]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.09241148084402084], [0.023195823654532433], [-0.002818223787471652], [0.026250513270497322], [0.1769580990076065], [0.013878539204597473]], dtype='float32').reshape([6, 1]),
            ]


    
    class PrimitiveOp_7cb14f960aeab4808b86a45135ef35c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 4116], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_27b5b128ba10267a935b42437507b859(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7cb14f960aeab4808b86a45135ef35c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24484822154045105]]], dtype='float32').reshape([1, 1, 1]),
            ]


    
    class PrimitiveOp_3f2a3497eea38b91547afba7aa59359c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 19, 34], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1145a817963464e8a8921462bc39cec1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f2a3497eea38b91547afba7aa59359c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0782fd68b15ee604fe265479d49807b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7385aec8a9e86495eec841db72d78256
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6e5e3d1d4343fe04d9a362d43301dee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c107832b4157be7086f9ee5ecf934921
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_728a21bbd1f6d9fa366153aa557034c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(60.11579132080078, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5ec1a7a8899513adacc9c616c4e7946b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(545.63427734375, dtype='float32').reshape([]),
                paddle.to_tensor([4.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_6ff4e60fa2f53236229a89e0cb8eddb9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2135, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2135, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1f13d05ae9951fe920985a331c3d088a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ff4e60fa2f53236229a89e0cb8eddb9
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f13d05ae9951fe920985a331c3d088a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ff4e60fa2f53236229a89e0cb8eddb9
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ac0ff7961357ee26274aff8fd8c5238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(-67143.1796875, dtype='float32').reshape([]),
                paddle.to_tensor([0.322158545255661], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3cbc50b5e569e937a5717539a48a9685(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(121165.015625, dtype='float32').reshape([]),
                paddle.to_tensor([0.322158545255661], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_c38c30d1e97bd9cd7d27f6613c7d7323(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 152, 272], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fbd4a25a4bc425b447969e69eb509291(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c38c30d1e97bd9cd7d27f6613c7d7323
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f7cdbefa946b309a7d0147b82f7ccc9c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_05b98f5312b8417f26cab5e75650a7b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7cdbefa946b309a7d0147b82f7ccc9c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_812308383c8137982374a4e7c9413569(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fc4318f494fa236721c6b350fe7bdf66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_812308383c8137982374a4e7c9413569
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3232676386833191], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_bc581670a35663b28fa08d7d536a5dd7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4590, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4590, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea297de86b9534b8ea5131068c849c33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc581670a35663b28fa08d7d536a5dd7
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea297de86b9534b8ea5131068c849c33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc581670a35663b28fa08d7d536a5dd7
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44a648f903e7f8de702819802de080f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(-729765.5, dtype='float32').reshape([]),
                paddle.to_tensor([0.204777792096138], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d35e7d97a863e80d2c61d8a0403ddf66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(261511.6875, dtype='float32').reshape([]),
                paddle.to_tensor([0.204777792096138], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64f985e809b4f8540738cab462a4b0fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cdac9f6ef2c4003ed867abb8507f929
        def get_inputs(self):
            return [
                paddle.to_tensor([305.69525146484375], dtype='float32').reshape([1]),
                paddle.to_tensor(2434.0, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_f024be2856cc89e51c3da2b05b1b9616(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1042, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1042, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_02143dae47f70bd96891410d774d6c7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f024be2856cc89e51c3da2b05b1b9616
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02143dae47f70bd96891410d774d6c7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f024be2856cc89e51c3da2b05b1b9616
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b6aabd607b8f60bb146bed6867c810e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(-19123.427734375, dtype='float32').reshape([]),
                paddle.to_tensor([0.3438152074813843], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b6fef4aa8e8dbbbf6b78a4388767f0e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(14883.05078125, dtype='float32').reshape([]),
                paddle.to_tensor([0.3438152074813843], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_61b7dfbd95a5b9e983a211e4e7ed9dd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b755566a0ec2e9481405ddcb54faa9da
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e4ed83efba059ddb11df4ba87689c65f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[100, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_75cf64bed9e7a7d0703d535385270b61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4ed83efba059ddb11df4ba87689c65f
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_35e9add6de32eccdde61cac6037f317c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_72aa0c802ad29d939380981cb600e5fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35e9add6de32eccdde61cac6037f317c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0020067794248461723, 0.2930905520915985, 0.2942022681236267, 0.4206882417201996], [0.022792115807533264, 0.4363101124763489, 0.3274344801902771, 0.033717453479766846]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.033379796892404556, 0.010689612478017807, 0.1018703281879425, 0.29704639315605164], [0.1340337097644806, 0.3128531277179718, 0.4129876494407654, 0.4426139295101166]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_e4d5e4dbeb753f9be751f37fe8979e6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae683b13bea09acdea6c3e1c0be2eab0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddecfc268c66168c9d6bc0cd786e0e57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_370b7b27371cf0eb63020a3f31061bd4
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([6069, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1f7522b0c13ba8944cf6300ba4e54447(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_282c6a51e828681db7572b4a0e9f705b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f7522b0c13ba8944cf6300ba4e54447
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f159d585ff7059843e351e75f586b19c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35e9add6de32eccdde61cac6037f317c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3937280774116516, 0.4412074089050293, 0.34095263481140137, 0.47780323028564453], [0.2755979895591736, 0.22411662340164185, 0.48828834295272827, 0.0838293507695198]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.34371617436408997, 0.4690535068511963, 0.36261194944381714, 0.16057159006595612], [0.02885439246892929, 0.22169719636440277, 0.21083924174308777, 0.35290396213531494]], dtype='float32').reshape([2, 4]),
            ]


    
    class PrimitiveOp_0adff6a8ed62bb1e41203249d6ff0731(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_77e1e513e646559aa83d85a91b1ca411(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0adff6a8ed62bb1e41203249d6ff0731
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.005169573239982128], [0.06563594937324524], [0.019044138491153717], [0.1293209046125412], [-0.01654825359582901]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_6c116731b106f52422b96d46a6e66477(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0adff6a8ed62bb1e41203249d6ff0731
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.017894238233566284], [0.021296389400959015], [0.09752587974071503], [-0.036712780594825745], [0.0270785354077816]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.012724664062261581], [0.08693233877420425], [0.11657001823186874], [0.09260812401771545], [0.010530282743275166]], dtype='float32').reshape([5, 1]),
            ]


    
    class PrimitiveOp_1582d163c6fb0022e25492784192f798(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 128, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9078453733b7866afe307d2f549537ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1582d163c6fb0022e25492784192f798
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3920208513736725], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8f2ae46f17671088fd9043666fea3299(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_636b2116e305caaee871cb19c45ea54a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6ef9f609328c6d1440af655d041f52e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_376de55309d118e1fd82e62f7f01348c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a0d6cdcd6c2e4a61984a64db15b1e33c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_29c315b38abf65cc88c86b1e30770a00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0d6cdcd6c2e4a61984a64db15b1e33c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_378b2323767f3042c5b8c305dcb17deb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2339, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2339, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c63f65b8310ef80a73fb8ccbe7d2ffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_378b2323767f3042c5b8c305dcb17deb
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c63f65b8310ef80a73fb8ccbe7d2ffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_378b2323767f3042c5b8c305dcb17deb
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a4a652e4bcf5ba0352c0e0c87223517(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(-11323224.0, dtype='float32').reshape([]),
                paddle.to_tensor([0.32941845059394836], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0b728e55b05777f6e345b7d0fcc776d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(133113.625, dtype='float32').reshape([]),
                paddle.to_tensor([0.32941845059394836], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_d88467fd12ba0e3ea964351e86357cd3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3063, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3063, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc8fa50f6f85ac1f56a44335bea73396(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d88467fd12ba0e3ea964351e86357cd3
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc8fa50f6f85ac1f56a44335bea73396(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d88467fd12ba0e3ea964351e86357cd3
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0403735881d8283f81f3880b128bca2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(193308.859375, dtype='float32').reshape([]),
                paddle.to_tensor([0.11696609854698181], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64b4cfd387ce13993dce1a0eca7213f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(174317.390625, dtype='float32').reshape([]),
                paddle.to_tensor([0.11696609854698181], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_2945a15ad4b1a75abd36295b05a9222b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3822, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3822, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a2cd31a4fa04f509dbdd384bb3246580(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2945a15ad4b1a75abd36295b05a9222b
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2cd31a4fa04f509dbdd384bb3246580(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2945a15ad4b1a75abd36295b05a9222b
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ce2593f1c5a03765c30effb321bbc53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(604421.6875, dtype='float32').reshape([]),
                paddle.to_tensor([0.19147957861423492], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1f862c7c7a47791be0581281d8280382(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(217196.53125, dtype='float32').reshape([]),
                paddle.to_tensor([0.19147957861423492], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_388cd211ba0040b84151d9f849c3e153(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_31463430a880cb0e7de5bfb473a34003(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_388cd211ba0040b84151d9f849c3e153
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.21355828642845154], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_05b98f5312b8417f26cab5e75650a7b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7cdbefa946b309a7d0147b82f7ccc9c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3553bbd1d75824fc96b05812bc4496f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(14.848526000976562, dtype='float32').reshape([]),
                paddle.to_tensor([3.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_dea8dd511af834402484677a1d04f743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c4f0ba0d4034fceae84d75338b628ae
        def get_inputs(self):
            return [
                paddle.uniform([20267, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([20267, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_17f1f74dd80dfa88292a6395ded48a7b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_045988d92015d8905ad1733aede578d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17f1f74dd80dfa88292a6395ded48a7b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.02299167960882187], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-0.04221170023083687], [0.14200380444526672], [-0.0011601647129282355], [-0.019290968775749207]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f706aebe0002f7bd99ff5a0a1367abd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17f1f74dd80dfa88292a6395ded48a7b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.053318336606025696], [0.01920013129711151], [0.008827432990074158], [0.08699016273021698]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.011106634512543678], [0.16120393574237823], [0.0076672681607306], [0.06769919395446777]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_609a88225f948a52adcb0408e17c279c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(4.3850321769714355, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_115b9f725461b0c490ef75e45a5820cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2057, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2057, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d747c0378b1c645db580fc532c28d3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_115b9f725461b0c490ef75e45a5820cd
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d747c0378b1c645db580fc532c28d3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_115b9f725461b0c490ef75e45a5820cd
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6f622dbc3ad19eeef1318dacf15b89c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(251715.625, dtype='float32').reshape([]),
                paddle.to_tensor([0.1501390039920807], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_30ae29e68edc5d2480a26101477ae2e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(29097.33984375, dtype='float32').reshape([]),
                paddle.to_tensor([0.1501390039920807], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_c7c88a4b0e6c9053ad407f332e761931(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b273796498a3973b896923beba799ca2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7c88a4b0e6c9053ad407f332e761931
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3421512842178345], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0db16e952a6e92dd7ab5e9f7f4a25e52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(33.76018142700195, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4c181afd1cd123a0e1b640ddb664e67e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_faf92cc7dea26277cdc049216c2d32ed
        def get_inputs(self):
            return [
                paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([6804, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29c315b38abf65cc88c86b1e30770a00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0d6cdcd6c2e4a61984a64db15b1e33c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74d1f477c02390b47f383c6b851920cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(237.84315490722656, dtype='float32').reshape([]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d098c5d5d912863417837240233691d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(138.18959045410156, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ff40957fd977a6a0d2571b5ff7ba8a44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98833d7b4c14a0e4c867033e684c3a13
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e5f2452ef0920e669a99ff56c695b8e8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7cc7c7f5f887db561b2e595ace5629b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f2452ef0920e669a99ff56c695b8e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2201fd9bd5aeaf2324583273dc21c64d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4189, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4189, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_251756529d93ee10280b11bd5642a341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2201fd9bd5aeaf2324583273dc21c64d
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_251756529d93ee10280b11bd5642a341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2201fd9bd5aeaf2324583273dc21c64d
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42fb2474dc8a38079af662accac6c0a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(47518.54296875, dtype='float32').reshape([]),
                paddle.to_tensor([0.3970640003681183], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_95fa920a1705a306ca6ef2199daa2661(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(238131.71875, dtype='float32').reshape([]),
                paddle.to_tensor([0.3970640003681183], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7cc7c7f5f887db561b2e595ace5629b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5f2452ef0920e669a99ff56c695b8e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f11be99b9ddb9b0670ad0cfa921480fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_398c814fc0b8c3ce8cb065fc6d81a4d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59bb48d3013aef44a2796ad8d1640f78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.23544012010097504]]], dtype='float32').reshape([1, 1, 1]),
            ]


    class TestPrimitiveOp_ab6df6369d71ef97592f8926a13663c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([12096, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a27c71a872ee36b9db66693cadc7729(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af94317360d7b61d66ac1d8a04bffc2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b6fd98b82b4150b1d44f7ed45e6ce47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_151713f40442abd0afb75fc9e3ad549a
        def get_inputs(self):
            return [
                paddle.to_tensor([1076.5631103515625], dtype='float32').reshape([1]),
                paddle.to_tensor(8732.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_c352c426ff9235df85ccf44de562513c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.00040858826832845807], [2.772476136669866e-06], [7.675283995922655e-05], [0.001655016327276826], [0.0001419818727299571], [0.018362853676080704]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_4fa3bf37f93cbba8f4684cbcb27e009e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[1.3540058716898784e-05], [1.6532496374566108e-05], [0.005603802856057882], [0.0017022115644067526], [0.010291928425431252], [0.0005788452690467238]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_5444ab3233bfe0cbc8c555a02ebd754e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.07576718181371689], [0.00447469437494874], [0.0977160707116127], [0.06420623511075974], [0.10864578187465668], [0.11122038960456848]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_8c3aa0947d47cd59e0d977811c557d9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_247550de1e8857b28f6a7e7f8e51b226(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f11be99b9ddb9b0670ad0cfa921480fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d77be0af403dcf7788ce7056ba84d11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23138ba4175e40da076559a9914ccb4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(8.992445945739746, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f5532aa571f5880479b5bbd8b77c10cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(2.4994864463806152, dtype='float32').reshape([]),
                paddle.to_tensor([2.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_af2f7c71c3e55f9b2398a3b6b365f747(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e7b38d269611da54609dde6631bba94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e7b38d269611da54609dde6631bba94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9be790f68a6a864a2768cd7745f90f69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(183327.234375, dtype='float32').reshape([]),
                paddle.to_tensor([0.42846009135246277], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_00a90afbb38e95d112fe346d8a10f955(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(97768.9921875, dtype='float32').reshape([]),
                paddle.to_tensor([0.42846009135246277], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4ba6e0ca6f3d4e7186af66db88a64437(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(940.4869384765625, dtype='float32').reshape([]),
                paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_95e715cd595f6f77cd7d5a58ecd6da29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c57f726ad15091268e7f32626ce4fb56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_661b6a4b0ba29b7cce96305573a6d40a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([5376, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad141b373c8b910e34512356a42e8677(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d889ca03446a56505675d2ca9c34c82d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32a9f90c8f1315312bff584b2a0bf8ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.010175604373216629], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.05231598764657974], [0.009299254976212978], [0.10837691277265549], [-0.07017733156681061], [0.07759939134120941], [-0.016110287979245186], [-0.0599064826965332], [0.08122386783361435], [-0.026601403951644897]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_5ca1050423aa4848e7e940950abb0419(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0013923496007919312], [0.029073655605316162], [-0.003807641565799713], [0.1609496772289276], [0.020773865282535553], [0.02696826308965683], [0.045601993799209595], [-0.08601643890142441], [0.10990434885025024]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.05092363804578781], [0.038372911512851715], [0.10456927120685577], [0.090772345662117], [0.09837325662374496], [0.010857976041734219], [-0.014304488897323608], [-0.004792569670826197], [0.08330294489860535]], dtype='float32').reshape([9, 1]),
            ]


    
    class PrimitiveOp_a3027de34ed145e471b26ca7e9684a54(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7c1987d47d60b68c3220391439554907(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3027de34ed145e471b26ca7e9684a54
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.12166859954595566], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8296e38f1e22d55391a4d4357a69e206(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8296e38f1e22d55391a4d4357a69e206(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c12bfa685ddf9243c72af4613d70fc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(32006.1796875, dtype='float32').reshape([]),
                paddle.to_tensor([0.10051140189170837], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_410357a450ec4fa17cd151a7acc1f4b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(3933.03173828125, dtype='float32').reshape([]),
                paddle.to_tensor([0.10051140189170837], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c594b7c5814e628652f0dd897c87ea8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, -0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
                paddle.to_tensor([0.020324425771832466, 0.013810954988002777, -0.004959273152053356, 0.040052562952041626, -0.04735404625535011, -0.022230884060263634], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_5f88f079b94c237fd71c59c171349645(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.014093050733208656, 0.05383152514696121, 0.031237879768013954, 0.005727421957999468, 0.03170868754386902, 0.040966905653476715], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0018855603411793709, 0.30608075857162476, 0.14130333065986633, 0.08980018645524979, 0.04460727050900459, 0.04448986425995827], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_2b8aa923331b21d5d594dd2f6f2e70e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.16486512124538422, 0.04520697519183159, -0.02553534507751465, -0.0034683942794799805, 0.11817975342273712, -0.3263065814971924], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.12327910959720612, 0.3055049479007721, 0.19421210885047913, -0.022199705243110657, -0.4006950855255127, 0.06812882423400879], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e4beee466ee4252bbda262c4765a0582(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.03921353816986084, 0.3743213713169098, 0.1907661110162735, 0.2556256353855133, 0.05274197459220886, -0.13855475187301636], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0158902108669281, -0.21657240390777588, -0.13378392159938812, 0.15638324618339539, -0.1620321273803711, 0.026508353650569916], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8e5a2c4b8af2638ce6f1a3e11bb5d9ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([1.812121868133545, 0.5770043730735779, 0.27817100286483765, 0.3044983446598053, 0.0003150637785438448, 0.00011431847815401852], dtype='float32').reshape([6]),
                paddle.to_tensor([2.812121868133545, 1.5770044326782227, 1.2781710624694824, 1.304498314857483, 1.0003150701522827, 1.0001143217086792], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e29f301f382da8856fe0bd9ee7bf52de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e29f301f382da8856fe0bd9ee7bf52de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b39865e92dc5c8ee48f3856609ef9b0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-1822.7001953125, dtype='float32').reshape([]),
                paddle.to_tensor([0.27057820558547974], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4e0c202fe799d1ca4173ab72375666c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(103763.359375, dtype='float32').reshape([]),
                paddle.to_tensor([0.27057820558547974], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_174b027b5bea4a74e1ba999470d656d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(958.8797607421875, dtype='float32').reshape([]),
                paddle.to_tensor([4.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0fc5bc1d540aca3224b55c7abc3c9226(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([8400, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cee15f2b2e7b40fded3a34c4bc8549ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c8b5622062799b7c46c030ddcbf2ff50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cee15f2b2e7b40fded3a34c4bc8549ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af94317360d7b61d66ac1d8a04bffc2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c57f726ad15091268e7f32626ce4fb56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b98d52efabd1eeef201cc34baae1c195(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b98d52efabd1eeef201cc34baae1c195(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d75af67acb11d9d294925910e1bb408(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-711917.6875, dtype='float32').reshape([]),
                paddle.to_tensor([0.21900232136249542], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6851299f29c9e7719330c974f9602a58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(87456.15625, dtype='float32').reshape([]),
                paddle.to_tensor([0.21900232136249542], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a1b2c6a66899a80e6a26e5e313354a4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.23765969276428223], [0.24743786454200745]]], dtype='float32').reshape([1, 2, 1]),
            ]


    class TestPrimitiveOp_247550de1e8857b28f6a7e7f8e51b226(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_398c814fc0b8c3ce8cb065fc6d81a4d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4da99e9a95721386fa7510d5e95d9d1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0011826036497950554]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_853d3085cebab6c1b16612821379758f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03306867927312851]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.03425128385424614]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_8f5b940a731d2ab5704202cbd17461e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.014378788881003857], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.24575075507164001], [-0.04205697402358055], [0.0034978860057890415], [0.003514286130666733], [0.1535715013742447], [0.1216726154088974]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_8ca0f8fc631031e2f36eef75ef177a57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.15333926677703857], [0.06525279581546783], [-0.00631610956043005], [0.02273622713983059], [0.023386597633361816], [-0.10779407620429993]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.09241148084402084], [0.023195823654532433], [-0.002818223787471652], [0.026250513270497322], [0.1769580990076065], [0.013878539204597473]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_4fbd374eaec5ecd8ec61682eab54b346(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24484822154045105]]], dtype='float32').reshape([1, 1, 1]),
            ]


    class TestPrimitiveOp_1da493b80fb47f4645ae154130dece34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cee15f2b2e7b40fded3a34c4bc8549ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95e715cd595f6f77cd7d5a58ecd6da29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a27c71a872ee36b9db66693cadc7729(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5432239bb8fb11b62e5977411ea1238c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(60.11579132080078, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_374bc874a4f085ff522ad9097525b045(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(545.63427734375, dtype='float32').reshape([]),
                paddle.to_tensor([4.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_70242b3f1759a3a59290067f9d3d4595(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70242b3f1759a3a59290067f9d3d4595(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_420ac1909858b9c4ef3fcd8cdbb6ad76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-67143.1796875, dtype='float32').reshape([]),
                paddle.to_tensor([0.322158545255661], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_22a9ae4ff1378e39353200acd270217d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(121165.015625, dtype='float32').reshape([]),
                paddle.to_tensor([0.322158545255661], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bd055b3497c1700d3dac406e47b2a0e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cee15f2b2e7b40fded3a34c4bc8549ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f17c4071a87bab74c185606e898b17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9889e38965aedf340444e51f9cf52a61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3027de34ed145e471b26ca7e9684a54
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3232676386833191], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4fb513c26dc067c253382f8bf638b7fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fb513c26dc067c253382f8bf638b7fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b380b1207d3bff2c4fbaaa8dc75d4cba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-729765.5, dtype='float32').reshape([]),
                paddle.to_tensor([0.204777792096138], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a2196964ea038e27a83ad97478f64f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(261511.6875, dtype='float32').reshape([]),
                paddle.to_tensor([0.204777792096138], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_751b315a0a607ef1137a72cc3b9d7e0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_151713f40442abd0afb75fc9e3ad549a
        def get_inputs(self):
            return [
                paddle.to_tensor([305.69525146484375], dtype='float32').reshape([1]),
                paddle.to_tensor(2434.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_b6c343ab8e11e7cd54d0040388a82a2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6c343ab8e11e7cd54d0040388a82a2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8be04bdc089ec0d4a331d2c7d81de553(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-19123.427734375, dtype='float32').reshape([]),
                paddle.to_tensor([0.3438152074813843], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_65ab386387cc112502819159e85cfd82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(14883.05078125, dtype='float32').reshape([]),
                paddle.to_tensor([0.3438152074813843], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8c3aa0947d47cd59e0d977811c557d9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b5913f714396094db06999c60510bd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f12ad5dcaea7d41bca1b5e27df5083e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0020067794248461723, 0.2930905520915985, 0.2942022681236267, 0.4206882417201996], [0.022792115807533264, 0.4363101124763489, 0.3274344801902771, 0.033717453479766846]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.033379796892404556, 0.010689612478017807, 0.1018703281879425, 0.29704639315605164], [0.1340337097644806, 0.3128531277179718, 0.4129876494407654, 0.4426139295101166]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_ad141b373c8b910e34512356a42e8677(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3790ca70aecdfcda3ebb84a5ef9246a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([6069, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f1b3cd85499ad40b2bbea47db117f846(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08a7636bc75ae2b6a71b588ab5d176f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3937280774116516, 0.4412074089050293, 0.34095263481140137, 0.47780323028564453], [0.2755979895591736, 0.22411662340164185, 0.48828834295272827, 0.0838293507695198]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.34371617436408997, 0.4690535068511963, 0.36261194944381714, 0.16057159006595612], [0.02885439246892929, 0.22169719636440277, 0.21083924174308777, 0.35290396213531494]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_816e8ae754700e7dc642d15724a8ed8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.005169573239982128], [0.06563594937324524], [0.019044138491153717], [0.1293209046125412], [-0.01654825359582901]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_1dbf3a943bf1a37f33c707836bae69be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.017894238233566284], [0.021296389400959015], [0.09752587974071503], [-0.036712780594825745], [0.0270785354077816]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.012724664062261581], [0.08693233877420425], [0.11657001823186874], [0.09260812401771545], [0.010530282743275166]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_2edfd561ccd25477d963736975bd6c3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3027de34ed145e471b26ca7e9684a54
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3920208513736725], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_af2f7c71c3e55f9b2398a3b6b365f747(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d889ca03446a56505675d2ca9c34c82d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_def5bbb48d4660d6672b91763f078ce8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3541530b304d8e551c164882db0dd3a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3541530b304d8e551c164882db0dd3a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83c93653f576564e74d3a8914b453660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-11323224.0, dtype='float32').reshape([]),
                paddle.to_tensor([0.32941845059394836], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9423e92934283d239c8b04adf5284f0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(133113.625, dtype='float32').reshape([]),
                paddle.to_tensor([0.32941845059394836], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b43bd569372b140972de1a3ce7a264e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b43bd569372b140972de1a3ce7a264e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81df1d32c0c915a6d5ef971579b49a7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(193308.859375, dtype='float32').reshape([]),
                paddle.to_tensor([0.11696609854698181], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_61ec547677c6ca3371444a9921e02ebc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(174317.390625, dtype='float32').reshape([]),
                paddle.to_tensor([0.11696609854698181], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f07d9c9d00803e72e5378ee90842ce46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f07d9c9d00803e72e5378ee90842ce46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d82639ad8a1ee7dc1644f7d09344c13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(604421.6875, dtype='float32').reshape([]),
                paddle.to_tensor([0.19147957861423492], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8ecdb9f4c3ecb5b4ee01c0f386b44d68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(217196.53125, dtype='float32').reshape([]),
                paddle.to_tensor([0.19147957861423492], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f3ad27d2f936e027a6754e5ba0c1eb26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3027de34ed145e471b26ca7e9684a54
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.21355828642845154], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9f17c4071a87bab74c185606e898b17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b336b4dbaf31bffcc1b90cafac29ef08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(14.848526000976562, dtype='float32').reshape([]),
                paddle.to_tensor([3.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_10ecd9d33778a2bd0ddf5676d9644c60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([20267, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([20267, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75fb3d314147a39880fcd7dc8c97ae54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.02299167960882187], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-0.04221170023083687], [0.14200380444526672], [-0.0011601647129282355], [-0.019290968775749207]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_94e0339b217dbecd4be18300f5cc1b9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.053318336606025696], [0.01920013129711151], [0.008827432990074158], [0.08699016273021698]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.011106634512543678], [0.16120393574237823], [0.0076672681607306], [0.06769919395446777]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f9454ebe1ac2c89c649ed852708923f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(4.3850321769714355, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d66f23484f26c430543242fb1eb5129f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d66f23484f26c430543242fb1eb5129f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27eb7713b7fba53189a2151a5e6e9fa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(251715.625, dtype='float32').reshape([]),
                paddle.to_tensor([0.1501390039920807], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5ff604726845472e068cc9eef187c073(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(29097.33984375, dtype='float32').reshape([]),
                paddle.to_tensor([0.1501390039920807], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_60d46632066bbb0ddf7e521fad38ecb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3027de34ed145e471b26ca7e9684a54
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3421512842178345], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4cba35131fbdf5210ee22eb97dc8db05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(33.76018142700195, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e7fefba9db86a9ff61ec644a21745415(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([6804, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_def5bbb48d4660d6672b91763f078ce8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d70de17803062049f8b6b540681f4bb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(237.84315490722656, dtype='float32').reshape([]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_836dd311c2f8230733a2d5269209b2e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(138.18959045410156, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2d77be0af403dcf7788ce7056ba84d11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79d79066cdddc014bcdadbf29f624cf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_614390ae10eb1f4a85de5c335946d2a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_614390ae10eb1f4a85de5c335946d2a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62ed930f8b205d567cec9c078c939206(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(47518.54296875, dtype='float32').reshape([]),
                paddle.to_tensor([0.3970640003681183], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_df13305f12f01d8b524aa5a05ba2ae9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(238131.71875, dtype='float32').reshape([]),
                paddle.to_tensor([0.3970640003681183], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_79d79066cdddc014bcdadbf29f624cf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()