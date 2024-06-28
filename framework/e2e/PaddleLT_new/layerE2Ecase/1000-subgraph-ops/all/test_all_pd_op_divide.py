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


    class TestPrimitiveOp_0e94bc4271aaa06eea578227a1d25bb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adf42b04d28c3f6e2d920614a9ca9273
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24457786977291107]]], dtype='float32').reshape([1, 1, 1]),
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


    class TestPrimitiveOp_172c3416bad05d27ab05482cf7761609(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_151713f40442abd0afb75fc9e3ad549a
        def get_inputs(self):
            return [
                paddle.to_tensor([1070.558837890625], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_c3b00e1745e76c37b4e8a6a6593345a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.00010075928730657324], [5.786511110272841e-07], [0.00029690199880860746], [0.004737387876957655], [0.022264447063207626], [0.004158198367804289]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_79c8ef464dca58440564ec712bebaf68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.005200754385441542], [0.009870069101452827], [0.00019271559722255915], [0.0014904242707416415], [0.0003329289029352367], [0.0008799833012744784]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_217633e344ed3a1fa42fb73b1fa1d2dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adf42b04d28c3f6e2d920614a9ca9273
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.040830276906490326], [0.2059045135974884], [0.007131705991923809], [0.2539832890033722], [0.10455213487148285], [0.02964458055794239]]], dtype='float32').reshape([1, 6, 1]),
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


    class TestPrimitiveOp_84a6c3b276f86a1a4e9e26781ee08d4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(9.116755485534668, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b7bf186e6479a062d696c45db9f43f21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(1.8068065643310547, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_7de12dc979070834c4de4cfd085cf6a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7de12dc979070834c4de4cfd085cf6a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a17185aade82c7f8240de113968d1e18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(40978.60546875, dtype='float32').reshape([]),
                paddle.to_tensor([0.23399683833122253], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ed018571f06491549f64ba6713fd09b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(100993.984375, dtype='float32').reshape([]),
                paddle.to_tensor([0.23399683833122253], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_590711f03d3b1dec84b422df2ac5abf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(952.041748046875, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_57522643a673dfd16276ee4c01c7712b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.012420357204973698], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.055676139891147614], [0.1168467253446579], [0.05836332216858864], [0.04395139962434769], [-0.006267378106713295], [0.004919853061437607], [-0.005880832672119141], [-0.006536051165312529], [0.1567060500383377]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_11384c976b5f85d30236fbcad43f21d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.034646838903427124], [0.02049916982650757], [0.02237202599644661], [-0.03931698575615883], [0.12258408963680267], [0.031192876398563385], [0.010459029115736485], [0.004638895858079195], [-0.12921343743801117]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.09032297879457474], [0.13734589517116547], [0.08073534816503525], [0.004634413868188858], [0.11631671339273453], [0.03611272946000099], [0.004578196443617344], [-0.0018971551908180118], [0.027492618188261986]], dtype='float32').reshape([9, 1]),
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


    class TestPrimitiveOp_2c825aeb238e87c538e5b875e96a2c71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f73f8a5a9cc5d1ba89df5ec87c9b6c49
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.014974748715758324], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2f387560e391b83c6b08ba748204f434(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f387560e391b83c6b08ba748204f434(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40a69e6c6f04d3922b3f39d6d795baf7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-40154.9140625, dtype='float32').reshape([]),
                paddle.to_tensor([0.11155679821968079], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7bb064afa23f9e3c1017b1052d4e27ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(3841.057373046875, dtype='float32').reshape([]),
                paddle.to_tensor([0.11155679821968079], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_42df8e80094eb64f662c20d0da300713(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.04189087823033333, 0.03793960437178612, -0.019696950912475586, -0.00199795956723392, -0.005094417370855808, -0.05107930675148964], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0288a4465694f5dd1ec13c316eb89442(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0031743552535772324, 0.1237947940826416, 0.04108375310897827, 0.0012921708403155208, 0.00019363850879017264, 0.016884468495845795], dtype='float32').reshape([6]),
                paddle.to_tensor([0.09627191722393036, 0.20486898720264435, 0.15970689058303833, 0.002337502548471093, 0.10713794827461243, 0.2054480016231537], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_ecc31b93f57d7dff6194d589d9f0deee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3056122064590454, 0.1645037829875946, -0.2240472137928009, 0.0053952038288116455, 0.25440096855163574, -0.1306624859571457], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.13707201182842255, 0.23063059151172638, 0.08791428804397583, -0.3703214228153229, -0.02002514898777008, 0.3909255564212799], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_99a15c68f27bffec7be3c9b0cb084414(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.18379609286785126, -0.09119249880313873, 0.15622317790985107, -0.12893515825271606, -0.2878800332546234, -0.4485952854156494], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.15612322092056274, -0.038688331842422485, -0.248960942029953, -0.41129761934280396, -0.33464959263801575, -0.2202756106853485], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_1f2257bff735c3d91113b8c0a810538b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.03235524147748947, 0.12258976697921753, 0.1641787737607956, 0.041074011474847794, 1.9663156270980835, 0.8367988467216492], dtype='float32').reshape([6]),
                paddle.to_tensor([1.0323551893234253, 1.1225898265838623, 1.164178729057312, 1.0410740375518799, 2.966315746307373, 1.836798906326294], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_86332c6596aec92dd15ab13575193ff4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86332c6596aec92dd15ab13575193ff4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1849d095364ce5cc4efa317575d53300(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(17140.609375, dtype='float32').reshape([]),
                paddle.to_tensor([0.2997754216194153], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_76c70197c5819cf014ad714cb5a03976(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(101553.640625, dtype='float32').reshape([]),
                paddle.to_tensor([0.2997754216194153], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_91696e22e90b5fb85a2b667e6ba0b330(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(943.4326171875, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_c60ff50addff1f67154d9966d7b67df2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c60ff50addff1f67154d9966d7b67df2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0a38939199b67d39877e94962be2ff6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(23171.12109375, dtype='float32').reshape([]),
                paddle.to_tensor([0.46540579199790955], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b877afcd1754041d43fb7d9e8ec0029a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(86169.0546875, dtype='float32').reshape([]),
                paddle.to_tensor([0.46540579199790955], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_79a363530bd0518222b70af98a9e083f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adf42b04d28c3f6e2d920614a9ca9273
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24594809114933014], [0.24552559852600098]]], dtype='float32').reshape([1, 2, 1]),
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


    class TestPrimitiveOp_55b06c6529e12baf3a5b29340912453d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.002530277008190751]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_9e28388d951f2091b2bda040431aa24b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15932731330394745]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.16185759007930756]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_75a65aa1d3ecc346deeb9b0a526b405f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.00018990683020092547], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-0.0036731057334691286], [0.035216521471738815], [0.10041127353906631], [-0.07269255071878433], [0.07398469001054764], [-0.0034735659137368202]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_f223095f63f6c1c7bed54675264bac27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08939338475465775], [-0.03296061232686043], [-0.11476945877075195], [0.05133692920207977], [0.061875276267528534], [0.03903312236070633]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.08572027832269669], [0.0022559084463864565], [-0.014358188025653362], [-0.02135562337934971], [0.13585996627807617], [0.035559557378292084]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_c78c7cee780eee92a0e46cf2cbbcb8a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adf42b04d28c3f6e2d920614a9ca9273
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.2478283792734146]]], dtype='float32').reshape([1, 1, 1]),
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


    class TestPrimitiveOp_fbbb0d7cf3af40361633c6d658aba7a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(61.51199722290039, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8650602dba65f75a5d746d0b3cdc6388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(553.7359619140625, dtype='float32').reshape([]),
                paddle.to_tensor([4.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e760deb346e69d3853b11928fd91d45a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e760deb346e69d3853b11928fd91d45a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7632e16a7af390e7acdc9c4f762362e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(72206.1484375, dtype='float32').reshape([]),
                paddle.to_tensor([0.00934692658483982], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b5fe57a7a3de1e1f5e918aee6de61e1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(121230.28125, dtype='float32').reshape([]),
                paddle.to_tensor([0.00934692658483982], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_880e1b0700d0546eb2e722c09fa76b88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f73f8a5a9cc5d1ba89df5ec87c9b6c49
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0019444238860160112], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8eab85d999c1e30c3cbff5c52ca6afaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8eab85d999c1e30c3cbff5c52ca6afaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f906b8700e55dfff3a9c1c411c00efe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-740155.875, dtype='float32').reshape([]),
                paddle.to_tensor([0.2666420042514801], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_366c6042c4d747f30e6791dc00bfb533(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(263422.34375, dtype='float32').reshape([]),
                paddle.to_tensor([0.2666420042514801], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_16d92b5ff5765d6ca2c25cadda577de0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_151713f40442abd0afb75fc9e3ad549a
        def get_inputs(self):
            return [
                paddle.to_tensor([296.0170593261719], dtype='float32').reshape([1]),
                paddle.to_tensor(2434.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_c9951cdf2fb68f8667370caf45c91d2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9951cdf2fb68f8667370caf45c91d2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_662bd2e995448784c7262779912459d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(2970.498046875, dtype='float32').reshape([]),
                paddle.to_tensor([0.135480597615242], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1bee45189a7a883de72add0848ad430c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(14721.1318359375, dtype='float32').reshape([]),
                paddle.to_tensor([0.135480597615242], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_9a9fbc8b826ebd4dbcbefd40df0455d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.35086989402770996, 0.40287554264068604, 0.21432794630527496, 0.2309100329875946], [0.36881744861602783, 0.044086262583732605, 0.3126998245716095, 0.49753546714782715]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.45365238189697266, 0.17781095206737518, 0.39713290333747864, 0.035417936742305756], [0.12685656547546387, 0.3015764653682709, 0.4608755111694336, 0.2108588069677353]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_4f64b3a08751f29956ae9b9de7669dba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.48548465967178345, 0.4446101188659668, 0.3393830358982086, 0.42001593112945557], [0.19515365362167358, 0.4153425991535187, 0.20683053135871887, 0.06365763396024704]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.3578750193119049, 0.1330009549856186, 0.31626880168914795, 0.4836221933364868], [0.04756021127104759, 0.19326059520244598, 0.3466707468032837, 0.33515799045562744]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_be80fd85a7527edffe6bbc1f30cc3fc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.06795760244131088], [0.04772914946079254], [-0.06907474249601364], [-0.05889728665351868], [0.017041902989149094]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_2d69f200ca16e55e37d2544a9cea8450(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03272394835948944], [0.010485194623470306], [0.08881095796823502], [0.016203150153160095], [-0.016797488555312157]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.03523365408182144], [0.05821434408426285], [0.019736213609576225], [-0.04269413650035858], [0.0002444145502522588]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_ab27577505beb0d83e291695861f45e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f73f8a5a9cc5d1ba89df5ec87c9b6c49
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.23169468343257904], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_cee010d96b74af7065a6b3ebf9960c9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cee010d96b74af7065a6b3ebf9960c9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_144e00f960258301279544780db15feb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(22387.4609375, dtype='float32').reshape([]),
                paddle.to_tensor([0.30249693989753723], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9cff889f2d694a61aac0963b38abf086(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(131671.375, dtype='float32').reshape([]),
                paddle.to_tensor([0.30249693989753723], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_67e555d5151a9bbf22965590b1816804(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67e555d5151a9bbf22965590b1816804(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bad2da0fe2628e332aec2597766f40d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-716387.5, dtype='float32').reshape([]),
                paddle.to_tensor([0.30826058983802795], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_368ceaee04ce47cb2e52ae456a5576cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(168724.640625, dtype='float32').reshape([]),
                paddle.to_tensor([0.30826058983802795], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4773f06ae107a90b25c3b1fb5cf83db4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4773f06ae107a90b25c3b1fb5cf83db4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f5a438e23a20daf77c868395c7facb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(38259.046875, dtype='float32').reshape([]),
                paddle.to_tensor([0.4110068678855896], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_eabdb913c38228fcf82d49322a4f1b15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(212590.09375, dtype='float32').reshape([]),
                paddle.to_tensor([0.4110068678855896], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7d9f9c560742898efffc9c9108dadde0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f73f8a5a9cc5d1ba89df5ec87c9b6c49
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.41867172718048096], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9f17c4071a87bab74c185606e898b17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce1481547e20a12bdc46166d304b2a89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(15.297125816345215, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_430cbf3c278fe3c41f981de790b994a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.09553957730531693], [-0.059163011610507965], [-0.05419258028268814], [0.01992667280137539]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_3585f44cc1dd12c409602765b2830ae5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.08313123881816864], [0.10774871706962585], [0.044438473880290985], [0.0210812296718359]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.012408342212438583], [0.04858570545911789], [-0.009754105471074581], [0.04100790247321129]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_58530c1b4b5b3c35faa42cb34884a5c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(4.511631965637207, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_020b0154b16ff8a25a21ca0faab47a02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_020b0154b16ff8a25a21ca0faab47a02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc13fea2a090154464d381b92241b733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(46659.546875, dtype='float32').reshape([]),
                paddle.to_tensor([0.4489540755748749], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_67c6722c4c039b0c37f69ebb6be1ceff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(28525.7109375, dtype='float32').reshape([]),
                paddle.to_tensor([0.4489540755748749], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_56922594cfd07b43e8584b0905661e4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f73f8a5a9cc5d1ba89df5ec87c9b6c49
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.17974981665611267], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_649aca199aeec577ba4c007b765ff767(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(35.15143585205078, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_d9da8f8fd9f084e03702399fbeb0fb03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(233.22496032714844, dtype='float32').reshape([]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cfe752bacdbe4990b65f3cc7281e4260(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(142.50640869140625, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_1be0e813ba673a59a8552b84fb60b8c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1be0e813ba673a59a8552b84fb60b8c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53a9e25009ed2c00ca1395f4d8c68965(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(166988.0625, dtype='float32').reshape([]),
                paddle.to_tensor([0.2619527280330658], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5c15fceaecd952d77b767a7d6a83f146(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(237993.203125, dtype='float32').reshape([]),
                paddle.to_tensor([0.2619527280330658], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_cdc767671cf5d045f941135d7021b736(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c905953f726f19f90c41ef4927424cef
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24457786977291107]]], dtype='float32').reshape([1, 1, 1]),
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


    class TestPrimitiveOp_75e0baba12c5abea031ce68c43f51074(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cdac9f6ef2c4003ed867abb8507f929
        def get_inputs(self):
            return [
                paddle.to_tensor([1070.558837890625], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_2c96ce613a82b6870cb3b6ec2a5b48bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca3dbde71d4cbf7e3bd6d9616fec7e12
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.00010075928730657324], [5.786511110272841e-07], [0.00029690199880860746], [0.004737387876957655], [0.022264447063207626], [0.004158198367804289]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_6059249da3c13f9a044a9c14955c82df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca3dbde71d4cbf7e3bd6d9616fec7e12
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.005200754385441542], [0.009870069101452827], [0.00019271559722255915], [0.0014904242707416415], [0.0003329289029352367], [0.0008799833012744784]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_4a0671ac7bcd4b910d006993e3f68a56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca3dbde71d4cbf7e3bd6d9616fec7e12
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.040830276906490326], [0.2059045135974884], [0.007131705991923809], [0.2539832890033722], [0.10455213487148285], [0.02964458055794239]]], dtype='float32').reshape([1, 6, 1]),
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


    class TestPrimitiveOp_f0a1059ace9c66ae231f136fab9741b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(9.116755485534668, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_743f72d146abbe31e24bc58facba58c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(1.8068065643310547, dtype='float32').reshape([]),
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


    
    class PrimitiveOp_4d5dccbe407e40755b5a16fcf5975ab6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_30260e7a8edc71f7cca05a988a0770be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d5dccbe407e40755b5a16fcf5975ab6
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30260e7a8edc71f7cca05a988a0770be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d5dccbe407e40755b5a16fcf5975ab6
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75b05899f8477e45d27e1b27470e1ff5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(40978.60546875, dtype='float32').reshape([]),
                paddle.to_tensor([0.23399683833122253], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_08fbcc4bb62e8aa2d390ff92dfe2aaaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(100993.984375, dtype='float32').reshape([]),
                paddle.to_tensor([0.23399683833122253], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c36c92928349dca6d365e1500170ad5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(952.041748046875, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_3da5af2d93a0486976b134aa1ef00dce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68ded890f87b9d3fd6c362b4fcfc0e1b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.012420357204973698], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.055676139891147614], [0.1168467253446579], [0.05836332216858864], [0.04395139962434769], [-0.006267378106713295], [0.004919853061437607], [-0.005880832672119141], [-0.006536051165312529], [0.1567060500383377]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_67965f599eb6e1525e23cc931500d42c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68ded890f87b9d3fd6c362b4fcfc0e1b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.034646838903427124], [0.02049916982650757], [0.02237202599644661], [-0.03931698575615883], [0.12258408963680267], [0.031192876398563385], [0.010459029115736485], [0.004638895858079195], [-0.12921343743801117]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.09032297879457474], [0.13734589517116547], [0.08073534816503525], [0.004634413868188858], [0.11631671339273453], [0.03611272946000099], [0.004578196443617344], [-0.0018971551908180118], [0.027492618188261986]], dtype='float32').reshape([9, 1]),
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


    class TestPrimitiveOp_ede762830c5df46deed90ebf69e9e4dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fea08d98e460778978a09aef3cd730f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.014974748715758324], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_cee15a99c99d128ea2025d3bfceffe44(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5454, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[5454, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_74462f11a3a8f84dc0b3c68b045a1f5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cee15a99c99d128ea2025d3bfceffe44
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74462f11a3a8f84dc0b3c68b045a1f5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cee15a99c99d128ea2025d3bfceffe44
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9b4097f29715e9c3facd559c690602c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(-40154.9140625, dtype='float32').reshape([]),
                paddle.to_tensor([0.11155679821968079], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_489620a454b81fff0a09e7c13f9c80ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(3841.057373046875, dtype='float32').reshape([]),
                paddle.to_tensor([0.11155679821968079], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_34fd031eaf529511e6a9a0c7b009ac6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_991840637b016f1ceba5f8049ac0c8ed
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.04189087823033333, 0.03793960437178612, -0.019696950912475586, -0.00199795956723392, -0.005094417370855808, -0.05107930675148964], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_de381aedbde6ee0c84531f334c0376f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_991840637b016f1ceba5f8049ac0c8ed
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0031743552535772324, 0.1237947940826416, 0.04108375310897827, 0.0012921708403155208, 0.00019363850879017264, 0.016884468495845795], dtype='float32').reshape([6]),
                paddle.to_tensor([0.09627191722393036, 0.20486898720264435, 0.15970689058303833, 0.002337502548471093, 0.10713794827461243, 0.2054480016231537], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_7ec21bb817adb696958745459f76ad92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_991840637b016f1ceba5f8049ac0c8ed
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3056122064590454, 0.1645037829875946, -0.2240472137928009, 0.0053952038288116455, 0.25440096855163574, -0.1306624859571457], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.13707201182842255, 0.23063059151172638, 0.08791428804397583, -0.3703214228153229, -0.02002514898777008, 0.3909255564212799], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_5d6f4c629c76a883c912f64f5ff10f23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_991840637b016f1ceba5f8049ac0c8ed
        def get_inputs(self):
            return [
                paddle.to_tensor([0.18379609286785126, -0.09119249880313873, 0.15622317790985107, -0.12893515825271606, -0.2878800332546234, -0.4485952854156494], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.15612322092056274, -0.038688331842422485, -0.248960942029953, -0.41129761934280396, -0.33464959263801575, -0.2202756106853485], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_04676d82704c2daf12f51d5dde347d68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_991840637b016f1ceba5f8049ac0c8ed
        def get_inputs(self):
            return [
                paddle.to_tensor([0.03235524147748947, 0.12258976697921753, 0.1641787737607956, 0.041074011474847794, 1.9663156270980835, 0.8367988467216492], dtype='float32').reshape([6]),
                paddle.to_tensor([1.0323551893234253, 1.1225898265838623, 1.164178729057312, 1.0410740375518799, 2.966315746307373, 1.836798906326294], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_79cf2e0360968e2350e8a2cdb3a98dc9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1722, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1722, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f0710fa1acd85f6b691a560547841c48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79cf2e0360968e2350e8a2cdb3a98dc9
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0710fa1acd85f6b691a560547841c48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79cf2e0360968e2350e8a2cdb3a98dc9
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9bae7b21dcc32e42fe22c259b348486(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(17140.609375, dtype='float32').reshape([]),
                paddle.to_tensor([0.2997754216194153], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_dcba0d1bfd319bc3b4ab598c015e8b1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(101553.640625, dtype='float32').reshape([]),
                paddle.to_tensor([0.2997754216194153], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bf5afb2b9af8ea597bfa141c313b946b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(943.4326171875, dtype='float32').reshape([]),
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


    
    class PrimitiveOp_1032d12678ca497e478416803e81bb70(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1518, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1518, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_24b947a163e2a13bf387b7b13b3c9801(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1032d12678ca497e478416803e81bb70
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24b947a163e2a13bf387b7b13b3c9801(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1032d12678ca497e478416803e81bb70
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5535f20f303b2edb28a6e452d4fb97fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(23171.12109375, dtype='float32').reshape([]),
                paddle.to_tensor([0.46540579199790955], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8a8c19707b7c4b81b9a2813fa978912e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(86169.0546875, dtype='float32').reshape([]),
                paddle.to_tensor([0.46540579199790955], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_133dc91a94a1e3105faba8638f52e062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d2e452b8e4207992fdbf0ad3db96a44
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24594809114933014], [0.24552559852600098]]], dtype='float32').reshape([1, 2, 1]),
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


    class TestPrimitiveOp_ada7b13ff764b9d40f66efe6ddabef71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e074e766a0ef5ac615b8de21d9514981
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.002530277008190751]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_55c02c67288b45ab13bedd9365d6596a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e074e766a0ef5ac615b8de21d9514981
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15932731330394745]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.16185759007930756]], dtype='float32').reshape([1, 1]),
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


    class TestPrimitiveOp_d0670145fda7e3c6ef8f30518f2eacf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d618a1b4a2ab12bf286cb1ad270ff0e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.00018990683020092547], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-0.0036731057334691286], [0.035216521471738815], [0.10041127353906631], [-0.07269255071878433], [0.07398469001054764], [-0.0034735659137368202]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_c713b9f9ee56b598f7a8ddf9272714c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d618a1b4a2ab12bf286cb1ad270ff0e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08939338475465775], [-0.03296061232686043], [-0.11476945877075195], [0.05133692920207977], [0.061875276267528534], [0.03903312236070633]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.08572027832269669], [0.0022559084463864565], [-0.014358188025653362], [-0.02135562337934971], [0.13585996627807617], [0.035559557378292084]], dtype='float32').reshape([6, 1]),
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


    class TestPrimitiveOp_4de0c722b2c40de63dd451ef8ae20f9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7cb14f960aeab4808b86a45135ef35c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.2478283792734146]]], dtype='float32').reshape([1, 1, 1]),
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


    class TestPrimitiveOp_0f14e45882c05589db84b769521c7a87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(61.51199722290039, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2b1f6b4f0fa4ee419f9dfe46615a608f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(553.7359619140625, dtype='float32').reshape([]),
                paddle.to_tensor([4.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_c3192f3a3a928cce2d6fa626aad81808(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2133, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2133, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f1ea936891a1500b7c7541296a13dd2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3192f3a3a928cce2d6fa626aad81808
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f1ea936891a1500b7c7541296a13dd2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3192f3a3a928cce2d6fa626aad81808
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16efc000b4b14c0849ac6c4f0b3085c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(72206.1484375, dtype='float32').reshape([]),
                paddle.to_tensor([0.00934692658483982], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5b6170c645185c6b49a61d33358df03b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(121230.28125, dtype='float32').reshape([]),
                paddle.to_tensor([0.00934692658483982], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_f5eb2759510571f0c824cafa92cdfff6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_812308383c8137982374a4e7c9413569
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0019444238860160112], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_c5a6ce4cce61c1ce9e5e7dd39d70fc31(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4631, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4631, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8b6fe72e407de125d8934de4341192bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5a6ce4cce61c1ce9e5e7dd39d70fc31
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b6fe72e407de125d8934de4341192bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5a6ce4cce61c1ce9e5e7dd39d70fc31
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5eee22b220fe6e0dc0d81817490742a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(-740155.875, dtype='float32').reshape([]),
                paddle.to_tensor([0.2666420042514801], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cbc9c741eb498d77c011839bea02ed20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(263422.34375, dtype='float32').reshape([]),
                paddle.to_tensor([0.2666420042514801], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6a7189f7e822ff12d1f2e9fb51b83b68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cdac9f6ef2c4003ed867abb8507f929
        def get_inputs(self):
            return [
                paddle.to_tensor([296.0170593261719], dtype='float32').reshape([1]),
                paddle.to_tensor(2434.0, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_468205f5efe599b16d3806d1f8ff4dea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1039, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1039, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6bd8b93c521c71825021a4be31fcfbf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_468205f5efe599b16d3806d1f8ff4dea
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bd8b93c521c71825021a4be31fcfbf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_468205f5efe599b16d3806d1f8ff4dea
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6392dfa3e3db801663eefe0c913217de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(2970.498046875, dtype='float32').reshape([]),
                paddle.to_tensor([0.135480597615242], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d24289a990c92f7c7c1b5246b9ea9842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(14721.1318359375, dtype='float32').reshape([]),
                paddle.to_tensor([0.135480597615242], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_acb2c9bc95036fbfff57da849e69ae03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35e9add6de32eccdde61cac6037f317c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.35086989402770996, 0.40287554264068604, 0.21432794630527496, 0.2309100329875946], [0.36881744861602783, 0.044086262583732605, 0.3126998245716095, 0.49753546714782715]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.45365238189697266, 0.17781095206737518, 0.39713290333747864, 0.035417936742305756], [0.12685656547546387, 0.3015764653682709, 0.4608755111694336, 0.2108588069677353]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_6a24c92d76458e24696d58731883770d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35e9add6de32eccdde61cac6037f317c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.48548465967178345, 0.4446101188659668, 0.3393830358982086, 0.42001593112945557], [0.19515365362167358, 0.4153425991535187, 0.20683053135871887, 0.06365763396024704]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.3578750193119049, 0.1330009549856186, 0.31626880168914795, 0.4836221933364868], [0.04756021127104759, 0.19326059520244598, 0.3466707468032837, 0.33515799045562744]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_c3716e8e9b5ee38ac98a573893839107(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0adff6a8ed62bb1e41203249d6ff0731
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.06795760244131088], [0.04772914946079254], [-0.06907474249601364], [-0.05889728665351868], [0.017041902989149094]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_8445fdd8dbd8b5cf35fb7cf0ccf4a324(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0adff6a8ed62bb1e41203249d6ff0731
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03272394835948944], [0.010485194623470306], [0.08881095796823502], [0.016203150153160095], [-0.016797488555312157]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.03523365408182144], [0.05821434408426285], [0.019736213609576225], [-0.04269413650035858], [0.0002444145502522588]], dtype='float32').reshape([5, 1]),
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


    class TestPrimitiveOp_d776c555e167824e66a6f10deeb6b1c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1582d163c6fb0022e25492784192f798
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.23169468343257904], dtype='float32').reshape([1]),
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


    
    class PrimitiveOp_c5f239c6f75be66c045c341b5c8e2b87(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2318, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2318, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_21f5a5a3e939a97a30c2474296e235b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5f239c6f75be66c045c341b5c8e2b87
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21f5a5a3e939a97a30c2474296e235b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5f239c6f75be66c045c341b5c8e2b87
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7a5a8c297eff7e11bf1e236dee15bac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(22387.4609375, dtype='float32').reshape([]),
                paddle.to_tensor([0.30249693989753723], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4525ff6eaf256c631740d1553f97e1e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(131671.375, dtype='float32').reshape([]),
                paddle.to_tensor([0.30249693989753723], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_a7cccd15e89baaca8ff500db13327c06(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2961, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2961, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f30d9b1585b13a5d55c50b7d5fe81ed4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7cccd15e89baaca8ff500db13327c06
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f30d9b1585b13a5d55c50b7d5fe81ed4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7cccd15e89baaca8ff500db13327c06
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa4a17dcc16cdb18df4a22b38ec97d3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(-716387.5, dtype='float32').reshape([]),
                paddle.to_tensor([0.30826058983802795], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_077d945e87316de9438af464f1a36a0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(168724.640625, dtype='float32').reshape([]),
                paddle.to_tensor([0.30826058983802795], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_6bddae2ec6c2b74a6ce23553769eb17e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3739, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3739, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b0d46449de2cce471d642311dc32aa11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bddae2ec6c2b74a6ce23553769eb17e
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0d46449de2cce471d642311dc32aa11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bddae2ec6c2b74a6ce23553769eb17e
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6983b7cfb47782e25c5932969b723801(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(38259.046875, dtype='float32').reshape([]),
                paddle.to_tensor([0.4110068678855896], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_88b7c4c13dc1393481e07a09301492cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(212590.09375, dtype='float32').reshape([]),
                paddle.to_tensor([0.4110068678855896], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_4656612e19b33a6fdfb86f8202a8c12c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_388cd211ba0040b84151d9f849c3e153
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.41867172718048096], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_05b98f5312b8417f26cab5e75650a7b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7cdbefa946b309a7d0147b82f7ccc9c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_366f4f5e78b08dc3f5cb87a32d07a124(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(15.297125816345215, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_30e1b727f8ae175785aa0abf99d4d9c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17f1f74dd80dfa88292a6395ded48a7b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.09553957730531693], [-0.059163011610507965], [-0.05419258028268814], [0.01992667280137539]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_95ef45faed4b0bcb52935515a1142391(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17f1f74dd80dfa88292a6395ded48a7b
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.08313123881816864], [0.10774871706962585], [0.044438473880290985], [0.0210812296718359]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.012408342212438583], [0.04858570545911789], [-0.009754105471074581], [0.04100790247321129]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_df96040f026b2db5be849d60ef46b880(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(4.511631965637207, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_14d31a32240b559b8d3bc09531e8d280(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2013, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2013, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7bbd61e9e2b13fb92ca40d4dde0e055d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14d31a32240b559b8d3bc09531e8d280
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bbd61e9e2b13fb92ca40d4dde0e055d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14d31a32240b559b8d3bc09531e8d280
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad8e7d0ac34e23c3a985ec9fbbbd268b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(46659.546875, dtype='float32').reshape([]),
                paddle.to_tensor([0.4489540755748749], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9a20adc80d7c95ac6a5a98b6e4876946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(28525.7109375, dtype='float32').reshape([]),
                paddle.to_tensor([0.4489540755748749], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_a4f6514dddca0072cdc4c17da5777962(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7c88a4b0e6c9053ad407f332e761931
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.17974981665611267], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6167fee3274921f7a779fc5a0f9fb491(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(35.15143585205078, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_5608d7b8497470bce42b3d225ecefc42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(233.22496032714844, dtype='float32').reshape([]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_dc10dd788ef4a6c11a23b0429740f650(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(142.50640869140625, dtype='float32').reshape([]),
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


    
    class PrimitiveOp_d0daf4e3cc899ca8d770c3c33b98b0da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4177, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4177, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c7139b9bea25c5b01031fdeb5584f54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0daf4e3cc899ca8d770c3c33b98b0da
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c7139b9bea25c5b01031fdeb5584f54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0daf4e3cc899ca8d770c3c33b98b0da
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dff569f6d9ef6c451d279fa3b0f972d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(166988.0625, dtype='float32').reshape([]),
                paddle.to_tensor([0.2619527280330658], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ec81fc197748e0c94e280665a23f3522(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(237993.203125, dtype='float32').reshape([]),
                paddle.to_tensor([0.2619527280330658], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_13819852327f5442bcbb3821bf2f2331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24457786977291107]]], dtype='float32').reshape([1, 1, 1]),
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


    class TestPrimitiveOp_172c3416bad05d27ab05482cf7761609(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_151713f40442abd0afb75fc9e3ad549a
        def get_inputs(self):
            return [
                paddle.to_tensor([1070.558837890625], dtype='float32').reshape([1]),
                paddle.to_tensor(8732.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_c3b00e1745e76c37b4e8a6a6593345a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.00010075928730657324], [5.786511110272841e-07], [0.00029690199880860746], [0.004737387876957655], [0.022264447063207626], [0.004158198367804289]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_79c8ef464dca58440564ec712bebaf68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.005200754385441542], [0.009870069101452827], [0.00019271559722255915], [0.0014904242707416415], [0.0003329289029352367], [0.0008799833012744784]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_2a003e6892c052250bf5a6928e57078e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.040830276906490326], [0.2059045135974884], [0.007131705991923809], [0.2539832890033722], [0.10455213487148285], [0.02964458055794239]]], dtype='float32').reshape([1, 6, 1]),
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


    class TestPrimitiveOp_84a6c3b276f86a1a4e9e26781ee08d4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(9.116755485534668, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b7bf186e6479a062d696c45db9f43f21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(1.8068065643310547, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_fcbbaac75545d67da184d6488f66832c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fcbbaac75545d67da184d6488f66832c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a17185aade82c7f8240de113968d1e18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(40978.60546875, dtype='float32').reshape([]),
                paddle.to_tensor([0.23399683833122253], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ed018571f06491549f64ba6713fd09b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(100993.984375, dtype='float32').reshape([]),
                paddle.to_tensor([0.23399683833122253], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_590711f03d3b1dec84b422df2ac5abf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(952.041748046875, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_57522643a673dfd16276ee4c01c7712b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.012420357204973698], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.055676139891147614], [0.1168467253446579], [0.05836332216858864], [0.04395139962434769], [-0.006267378106713295], [0.004919853061437607], [-0.005880832672119141], [-0.006536051165312529], [0.1567060500383377]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_11384c976b5f85d30236fbcad43f21d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.034646838903427124], [0.02049916982650757], [0.02237202599644661], [-0.03931698575615883], [0.12258408963680267], [0.031192876398563385], [0.010459029115736485], [0.004638895858079195], [-0.12921343743801117]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.09032297879457474], [0.13734589517116547], [0.08073534816503525], [0.004634413868188858], [0.11631671339273453], [0.03611272946000099], [0.004578196443617344], [-0.0018971551908180118], [0.027492618188261986]], dtype='float32').reshape([9, 1]),
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


    class TestPrimitiveOp_91141f96d542c6bd8ad73f39df2f9cd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3027de34ed145e471b26ca7e9684a54
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.014974748715758324], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0894c4e2bbfa7e5376482ad01ea00c3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0894c4e2bbfa7e5376482ad01ea00c3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40a69e6c6f04d3922b3f39d6d795baf7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-40154.9140625, dtype='float32').reshape([]),
                paddle.to_tensor([0.11155679821968079], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7bb064afa23f9e3c1017b1052d4e27ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(3841.057373046875, dtype='float32').reshape([]),
                paddle.to_tensor([0.11155679821968079], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_42df8e80094eb64f662c20d0da300713(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.04189087823033333, 0.03793960437178612, -0.019696950912475586, -0.00199795956723392, -0.005094417370855808, -0.05107930675148964], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0288a4465694f5dd1ec13c316eb89442(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0031743552535772324, 0.1237947940826416, 0.04108375310897827, 0.0012921708403155208, 0.00019363850879017264, 0.016884468495845795], dtype='float32').reshape([6]),
                paddle.to_tensor([0.09627191722393036, 0.20486898720264435, 0.15970689058303833, 0.002337502548471093, 0.10713794827461243, 0.2054480016231537], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_ecc31b93f57d7dff6194d589d9f0deee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3056122064590454, 0.1645037829875946, -0.2240472137928009, 0.0053952038288116455, 0.25440096855163574, -0.1306624859571457], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.13707201182842255, 0.23063059151172638, 0.08791428804397583, -0.3703214228153229, -0.02002514898777008, 0.3909255564212799], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_99a15c68f27bffec7be3c9b0cb084414(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.18379609286785126, -0.09119249880313873, 0.15622317790985107, -0.12893515825271606, -0.2878800332546234, -0.4485952854156494], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.15612322092056274, -0.038688331842422485, -0.248960942029953, -0.41129761934280396, -0.33464959263801575, -0.2202756106853485], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_1f2257bff735c3d91113b8c0a810538b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.03235524147748947, 0.12258976697921753, 0.1641787737607956, 0.041074011474847794, 1.9663156270980835, 0.8367988467216492], dtype='float32').reshape([6]),
                paddle.to_tensor([1.0323551893234253, 1.1225898265838623, 1.164178729057312, 1.0410740375518799, 2.966315746307373, 1.836798906326294], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c43cc70b264fcb3ab3a2752c62590039(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c43cc70b264fcb3ab3a2752c62590039(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1849d095364ce5cc4efa317575d53300(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(17140.609375, dtype='float32').reshape([]),
                paddle.to_tensor([0.2997754216194153], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_76c70197c5819cf014ad714cb5a03976(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(101553.640625, dtype='float32').reshape([]),
                paddle.to_tensor([0.2997754216194153], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_91696e22e90b5fb85a2b667e6ba0b330(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(943.4326171875, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_cad7f80937e718e0373c0010382aa64f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cad7f80937e718e0373c0010382aa64f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0a38939199b67d39877e94962be2ff6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(23171.12109375, dtype='float32').reshape([]),
                paddle.to_tensor([0.46540579199790955], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b877afcd1754041d43fb7d9e8ec0029a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(86169.0546875, dtype='float32').reshape([]),
                paddle.to_tensor([0.46540579199790955], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_19c104a396a374e8530f71d6bf0f5d06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24594809114933014], [0.24552559852600098]]], dtype='float32').reshape([1, 2, 1]),
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


    class TestPrimitiveOp_55b06c6529e12baf3a5b29340912453d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.002530277008190751]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_9e28388d951f2091b2bda040431aa24b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15932731330394745]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.16185759007930756]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_75a65aa1d3ecc346deeb9b0a526b405f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.00018990683020092547], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-0.0036731057334691286], [0.035216521471738815], [0.10041127353906631], [-0.07269255071878433], [0.07398469001054764], [-0.0034735659137368202]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_f223095f63f6c1c7bed54675264bac27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08939338475465775], [-0.03296061232686043], [-0.11476945877075195], [0.05133692920207977], [0.061875276267528534], [0.03903312236070633]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.08572027832269669], [0.0022559084463864565], [-0.014358188025653362], [-0.02135562337934971], [0.13585996627807617], [0.035559557378292084]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_04cf9c632e777f242736cff6cd5d0c85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.2478283792734146]]], dtype='float32').reshape([1, 1, 1]),
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


    class TestPrimitiveOp_fbbb0d7cf3af40361633c6d658aba7a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(61.51199722290039, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8650602dba65f75a5d746d0b3cdc6388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(553.7359619140625, dtype='float32').reshape([]),
                paddle.to_tensor([4.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c32b571e4f08d79e3201cb712db8a75e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c32b571e4f08d79e3201cb712db8a75e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7632e16a7af390e7acdc9c4f762362e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(72206.1484375, dtype='float32').reshape([]),
                paddle.to_tensor([0.00934692658483982], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b5fe57a7a3de1e1f5e918aee6de61e1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(121230.28125, dtype='float32').reshape([]),
                paddle.to_tensor([0.00934692658483982], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_9a4081d5f37b76a343d6db04207036fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3027de34ed145e471b26ca7e9684a54
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0019444238860160112], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4d1c8dea1af164576cea66fff3c2d37a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d1c8dea1af164576cea66fff3c2d37a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f906b8700e55dfff3a9c1c411c00efe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-740155.875, dtype='float32').reshape([]),
                paddle.to_tensor([0.2666420042514801], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_366c6042c4d747f30e6791dc00bfb533(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(263422.34375, dtype='float32').reshape([]),
                paddle.to_tensor([0.2666420042514801], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_16d92b5ff5765d6ca2c25cadda577de0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_151713f40442abd0afb75fc9e3ad549a
        def get_inputs(self):
            return [
                paddle.to_tensor([296.0170593261719], dtype='float32').reshape([1]),
                paddle.to_tensor(2434.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_0a4f30689dc776211b5a5269820eb73d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a4f30689dc776211b5a5269820eb73d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_662bd2e995448784c7262779912459d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(2970.498046875, dtype='float32').reshape([]),
                paddle.to_tensor([0.135480597615242], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1bee45189a7a883de72add0848ad430c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(14721.1318359375, dtype='float32').reshape([]),
                paddle.to_tensor([0.135480597615242], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_9a9fbc8b826ebd4dbcbefd40df0455d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.35086989402770996, 0.40287554264068604, 0.21432794630527496, 0.2309100329875946], [0.36881744861602783, 0.044086262583732605, 0.3126998245716095, 0.49753546714782715]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.45365238189697266, 0.17781095206737518, 0.39713290333747864, 0.035417936742305756], [0.12685656547546387, 0.3015764653682709, 0.4608755111694336, 0.2108588069677353]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_4f64b3a08751f29956ae9b9de7669dba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.48548465967178345, 0.4446101188659668, 0.3393830358982086, 0.42001593112945557], [0.19515365362167358, 0.4153425991535187, 0.20683053135871887, 0.06365763396024704]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.3578750193119049, 0.1330009549856186, 0.31626880168914795, 0.4836221933364868], [0.04756021127104759, 0.19326059520244598, 0.3466707468032837, 0.33515799045562744]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_be80fd85a7527edffe6bbc1f30cc3fc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.06795760244131088], [0.04772914946079254], [-0.06907474249601364], [-0.05889728665351868], [0.017041902989149094]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_2d69f200ca16e55e37d2544a9cea8450(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03272394835948944], [0.010485194623470306], [0.08881095796823502], [0.016203150153160095], [-0.016797488555312157]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.03523365408182144], [0.05821434408426285], [0.019736213609576225], [-0.04269413650035858], [0.0002444145502522588]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_0710c7f4bba8cf3d18e94e63fa002e04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3027de34ed145e471b26ca7e9684a54
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.23169468343257904], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_a262bc38f7e05c13cdf1218e851dcb34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a262bc38f7e05c13cdf1218e851dcb34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_144e00f960258301279544780db15feb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(22387.4609375, dtype='float32').reshape([]),
                paddle.to_tensor([0.30249693989753723], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9cff889f2d694a61aac0963b38abf086(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(131671.375, dtype='float32').reshape([]),
                paddle.to_tensor([0.30249693989753723], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4bb567bd0fc9c0149891fc54bf122335(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bb567bd0fc9c0149891fc54bf122335(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bad2da0fe2628e332aec2597766f40d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-716387.5, dtype='float32').reshape([]),
                paddle.to_tensor([0.30826058983802795], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_368ceaee04ce47cb2e52ae456a5576cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(168724.640625, dtype='float32').reshape([]),
                paddle.to_tensor([0.30826058983802795], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_813d74e4d3b2f1c33e62378a79cfb7df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_813d74e4d3b2f1c33e62378a79cfb7df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f5a438e23a20daf77c868395c7facb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(38259.046875, dtype='float32').reshape([]),
                paddle.to_tensor([0.4110068678855896], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_eabdb913c38228fcf82d49322a4f1b15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(212590.09375, dtype='float32').reshape([]),
                paddle.to_tensor([0.4110068678855896], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b35ce4d1f57efc307a00a9e86240785d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3027de34ed145e471b26ca7e9684a54
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.41867172718048096], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9f17c4071a87bab74c185606e898b17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce1481547e20a12bdc46166d304b2a89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(15.297125816345215, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_430cbf3c278fe3c41f981de790b994a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.09553957730531693], [-0.059163011610507965], [-0.05419258028268814], [0.01992667280137539]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_3585f44cc1dd12c409602765b2830ae5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.08313123881816864], [0.10774871706962585], [0.044438473880290985], [0.0210812296718359]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.012408342212438583], [0.04858570545911789], [-0.009754105471074581], [0.04100790247321129]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_58530c1b4b5b3c35faa42cb34884a5c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(4.511631965637207, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_67ec4c2cc54f2330f8bd9c7c9a086814(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67ec4c2cc54f2330f8bd9c7c9a086814(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc13fea2a090154464d381b92241b733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(46659.546875, dtype='float32').reshape([]),
                paddle.to_tensor([0.4489540755748749], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_67c6722c4c039b0c37f69ebb6be1ceff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(28525.7109375, dtype='float32').reshape([]),
                paddle.to_tensor([0.4489540755748749], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ebf72cb85e113bc0420d85f226a5ab5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3027de34ed145e471b26ca7e9684a54
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.17974981665611267], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_649aca199aeec577ba4c007b765ff767(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(35.15143585205078, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_d9da8f8fd9f084e03702399fbeb0fb03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(233.22496032714844, dtype='float32').reshape([]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cfe752bacdbe4990b65f3cc7281e4260(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(142.50640869140625, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_60bdfb65714239df023dbb1660d85a3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60bdfb65714239df023dbb1660d85a3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53a9e25009ed2c00ca1395f4d8c68965(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(166988.0625, dtype='float32').reshape([]),
                paddle.to_tensor([0.2619527280330658], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5c15fceaecd952d77b767a7d6a83f146(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(237993.203125, dtype='float32').reshape([]),
                paddle.to_tensor([0.2619527280330658], dtype='float32').reshape([1]),
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