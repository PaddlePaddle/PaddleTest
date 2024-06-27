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


    class TestPrimitiveOp_083056651d63eda6c0c6187557767030(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adf42b04d28c3f6e2d920614a9ca9273
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24608883261680603]]], dtype='float32').reshape([1, 1, 1]),
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


    class TestPrimitiveOp_2c436f2187ef9e207e6a62ccb0d39405(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_151713f40442abd0afb75fc9e3ad549a
        def get_inputs(self):
            return [
                paddle.to_tensor([1110.6275634765625], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_583d237ac57c4b7666406fffec30da37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.0003679796354845166], [0.0011984009761363268], [0.0008312652935273945], [0.0037753786891698837], [0.01995982974767685], [0.00011072401684941724]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_52c6d0fdb0d33bb278d4dbec58409848(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.001364782452583313], [0.0004077347693964839], [0.030202772468328476], [0.0028741590213030577], [0.026446755975484848], [0.0006722644320689142]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_21204bc726b70630f1808d8048901cd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adf42b04d28c3f6e2d920614a9ca9273
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.05796103551983833], [0.05304250493645668], [0.16478216648101807], [0.05082429572939873], [0.16708996891975403], [0.05151817202568054]]], dtype='float32').reshape([1, 6, 1]),
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


    class TestPrimitiveOp_9ae83e98e566ab50d65d440a0060f1fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(9.806840896606445, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7efa70b7be76f3172370a367d323def1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(3.564072608947754, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_f2a1c39da2efab25a3a9dd67d31321e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2a1c39da2efab25a3a9dd67d31321e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b51910e10a979753104572d7ad6c2777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-449226.28125, dtype='float32').reshape([]),
                paddle.to_tensor([0.49678489565849304], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_06c9f545b441c5cb19083c97be161b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(96541.46875, dtype='float32').reshape([]),
                paddle.to_tensor([0.49678489565849304], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5a202f8c366968da80dc3f590d798e9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(953.4589233398438, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_dca7ef59040627ce9a44168838ea470f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.02909252792596817], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[-0.08513391762971878], [-0.0048834290355443954], [0.09082716703414917], [0.06779509782791138], [-0.02920367568731308], [0.020875904709100723], [0.09581932425498962], [-0.005090379621833563], [0.07913517206907272]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_19f62d0c69e16998629901b2d1428b30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.11926756799221039], [0.008719295263290405], [-0.030501816421747208], [-0.02372565120458603], [0.010544508695602417], [-0.05020035803318024], [0.0], [0.05388128384947777], [-0.07870669662952423]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.034133654087781906], [0.0038358664605766535], [0.06032535061240196], [0.04406944662332535], [-0.018659166991710663], [-0.029324453324079514], [0.09581932425498962], [0.04879090562462807], [0.0004284780006855726]], dtype='float32').reshape([9, 1]),
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


    class TestPrimitiveOp_e5a0950dadb10f80210bb72fc04233be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f73f8a5a9cc5d1ba89df5ec87c9b6c49
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3117090165615082], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a96ec07cb14a89f5f351cd378cd65c0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a96ec07cb14a89f5f351cd378cd65c0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea19c67bdccbce1d84dce2514d5a3a4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-207.77392578125, dtype='float32').reshape([]),
                paddle.to_tensor([0.10024213790893555], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_740b2fd44fd7877443aaa75c5f52d1c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(3922.744140625, dtype='float32').reshape([]),
                paddle.to_tensor([0.10024213790893555], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_b02a61270ce5a82d16f8175b7789acab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.0, 0.0, -0.0, 0.0, 0.0, -0.0], dtype='float32').reshape([6]),
                paddle.to_tensor([0.07526008784770966, 0.02078755758702755, -0.02064768597483635, 0.003164077177643776, 0.009952809661626816, 0.030713750049471855], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_eaf4e1ee195261f4646cb6183770752b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.07436105608940125, 0.038930028676986694, 0.004602316301316023, 0.08816459029912949, 0.04051024839282036, 0.03402898088097572], dtype='float32').reshape([6]),
                paddle.to_tensor([0.26359811425209045, 0.0007747879717499018, 0.06292464584112167, 0.14202694594860077, 0.04369882866740227, 0.15298713743686676], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_6addad4c03917cd3f79f98d0f5fc280b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.002785980701446533, -0.30777591466903687, 0.25031355023384094, 0.015211403369903564, -0.05733698606491089, 0.1896750032901764], dtype='float32').reshape([6]),
                paddle.to_tensor([0.20420202612876892, -0.0675412118434906, -0.08248728513717651, 0.20800691843032837, -0.1735844612121582, 0.16192829608917236], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_d4d9a01ff5df415fed9490836fd9d705(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.25374525785446167, -0.32804441452026367, -0.25219541788101196, -0.3383956551551819, -0.24549104273319244, -0.06895255297422409], dtype='float32').reshape([6]),
                paddle.to_tensor([0.29435500502586365, -0.17794977128505707, 0.016363710165023804, -0.21813932061195374, 0.1977015733718872, 0.17290432751178741], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_46e6d18fb34285df320930f1e474b8dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.19734278321266174, 0.03200355917215347, 0.026051675900816917, 0.34692761301994324, 0.5951855778694153, 0.6268026828765869], dtype='float32').reshape([6]),
                paddle.to_tensor([1.1973427534103394, 1.0320035219192505, 1.026051640510559, 1.3469276428222656, 1.5951855182647705, 1.626802682876587], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_d521e5bdeeeee5773e8231cfb5868e00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d521e5bdeeeee5773e8231cfb5868e00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75e5cf69da0bbf9728ce2ab839624610(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(211035.953125, dtype='float32').reshape([]),
                paddle.to_tensor([0.3157075345516205], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_732331af0ac8476bfaf654f4bd13e3bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(106355.296875, dtype='float32').reshape([]),
                paddle.to_tensor([0.3157075345516205], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c95c52c0d7dd4f045b003236df817652(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(948.6329345703125, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_052771d79220c185c6f85804cd0a87b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_052771d79220c185c6f85804cd0a87b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea64e8804335e022f58943bde2bba9f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(28115.0546875, dtype='float32').reshape([]),
                paddle.to_tensor([0.28394192457199097], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7e7da7232e278bdbf31b08687fc86f03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(85368.09375, dtype='float32').reshape([]),
                paddle.to_tensor([0.28394192457199097], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7b952e3e86285cbc4774db62ac8632e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adf42b04d28c3f6e2d920614a9ca9273
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.2430521547794342], [0.24500004947185516]]], dtype='float32').reshape([1, 2, 1]),
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


    class TestPrimitiveOp_51add1ce07ba6a789eb220bded02f00f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0809708684682846]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_d8c1b37093abae1681a7866e83e099bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.006399311125278473]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.08737017959356308]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_6b3131d3f47f5d5832a962476218308f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-0.009171975776553154], [-0.0004534857871476561], [-0.13099893927574158], [-0.018913721665740013], [-0.02232457511126995], [0.0065928734838962555]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_1377573c5cc54d8a1428a5a9654ca85d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.026915011927485466], [0.03408973291516304], [0.1773102730512619], [0.013566361740231514], [0.0009194985032081604], [-0.011294983327388763]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.017743036150932312], [0.033636245876550674], [0.046311333775520325], [-0.0053473603911697865], [-0.02140507660806179], [-0.004702110309153795]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_fa0d0a40b5f472a0032d8b37995159f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adf42b04d28c3f6e2d920614a9ca9273
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.2399139702320099]]], dtype='float32').reshape([1, 1, 1]),
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


    class TestPrimitiveOp_52c0ae16588325a46e71e70206df662a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(58.3995246887207, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ac59ec19e81ad2ebc687900b9075ed57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(559.1926879882812, dtype='float32').reshape([]),
                paddle.to_tensor([4.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a74da698438067c718f3277b8315dd44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a74da698438067c718f3277b8315dd44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef6d7a260ea383091c139875e2b48020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(66534.359375, dtype='float32').reshape([]),
                paddle.to_tensor([0.38607752323150635], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6fd18195dd7997aed5375c6f0fb2b80a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(116012.0546875, dtype='float32').reshape([]),
                paddle.to_tensor([0.38607752323150635], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_4047b20ad69f0568534c63bffb3607e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f73f8a5a9cc5d1ba89df5ec87c9b6c49
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.26358479261398315], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ebbdcef54da2b25de33538b385496031(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebbdcef54da2b25de33538b385496031(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20a7df03dc523325dc2ea8a956188c9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(125483.75, dtype='float32').reshape([]),
                paddle.to_tensor([0.004646401386708021], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a9b23b48638404f356c2e6493dd4f1c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(260588.625, dtype='float32').reshape([]),
                paddle.to_tensor([0.004646401386708021], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_73b32a76a318e2a3dfc84b227c557055(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_151713f40442abd0afb75fc9e3ad549a
        def get_inputs(self):
            return [
                paddle.to_tensor([316.174560546875], dtype='float32').reshape([1]),
                paddle.to_tensor(2434.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_dc858a92e28c7469c2afeeb814f35453(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc858a92e28c7469c2afeeb814f35453(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0edc65dfdc6d0b015aa9c308337ba33d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(63740.50390625, dtype='float32').reshape([]),
                paddle.to_tensor([0.04321810603141785], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9048c9091e2da1ee790edd4496f37842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(15261.7607421875, dtype='float32').reshape([]),
                paddle.to_tensor([0.04321810603141785], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_fff6fe2903691518dbb3572e1e5db7a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4150143563747406, 0.17286737263202667, 0.18879666924476624, 0.00761602446436882], [0.22656339406967163, 0.46918755769729614, 0.38365957140922546, 0.12311527878046036]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.30250710248947144, 0.2891940772533417, 0.37603989243507385, 0.36301806569099426], [0.42629826068878174, 0.40945640206336975, 0.46440133452415466, 0.19835348427295685]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_8c1e277c8fc9095b4d422a75607c0b19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17972084879875183, 0.021701132878661156, 0.0599205382168293, 0.38248834013938904], [0.13160006701946259, 0.4700474739074707, 0.3898284435272217, 0.3776264488697052]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.45377030968666077, 0.47708168625831604, 0.3173177242279053, 0.03736255690455437], [0.439419686794281, 0.11878792941570282, 0.030491745099425316, 0.06279624998569489]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_7c1e390d66a3a02e238a8284cac699c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.015536606311798096], [-0.028670985251665115], [0.08992450684309006], [-0.0018401051638647914], [0.0011996077373623848]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_01a8227f6b620a42b64da3fa1734c42f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0403393991291523], [0.11921769380569458], [0.02897804230451584], [-0.00499636959284544], [0.0033686147071421146]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.024802792817354202], [0.09054671227931976], [0.1189025491476059], [-0.006836474873125553], [0.0045682224445044994]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_f5f34f6a829f9114f1aa8d9a9543b62b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f73f8a5a9cc5d1ba89df5ec87c9b6c49
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.2312423139810562], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_1ff4546f4530c71c07715a23e9777e9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ff4546f4530c71c07715a23e9777e9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4dae842277328be20948c2ae90695e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(154219.03125, dtype='float32').reshape([]),
                paddle.to_tensor([0.0921860933303833], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_75736d9b71fa1ce2187b83b41e602ad9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(134749.84375, dtype='float32').reshape([]),
                paddle.to_tensor([0.0921860933303833], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e42f1cc1def7c8dce3e5cf8bf0aee086(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e42f1cc1def7c8dce3e5cf8bf0aee086(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96e0049ddc974702c8d20e8e92d1c6c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(364645.03125, dtype='float32').reshape([]),
                paddle.to_tensor([0.4369841516017914], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9485bad0b7d48672823e6e02ba9d0974(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(170197.265625, dtype='float32').reshape([]),
                paddle.to_tensor([0.4369841516017914], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c310d0a812919f727eff25acf6462247(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c310d0a812919f727eff25acf6462247(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3369327041e6ff9f6fc42ab0de5be200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-1592628.125, dtype='float32').reshape([]),
                paddle.to_tensor([0.05049596726894379], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_14d3ea0960d9e3fa8e0bd1767e78ac00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(217823.0, dtype='float32').reshape([]),
                paddle.to_tensor([0.05049596726894379], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_709e31147c8acf8ebdb6728cdeb37a7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f73f8a5a9cc5d1ba89df5ec87c9b6c49
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.04804033041000366], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9f17c4071a87bab74c185606e898b17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_724fecb829d8f33326a9cb3444f219b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(14.4016695022583, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_91bcbe803f31bf03ddbaca2c7a0ac7ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-0.06323745846748352], [-0.025388313457369804], [0.017503943294286728], [0.007693326100707054]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_4c06d175098e85308eabcf7194949cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08949588239192963], [0.03676968067884445], [0.012986956164240837], [-0.008089913055300713]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.026258420199155807], [0.011381367221474648], [0.030490899458527565], [-0.0003965869836974889]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_a600e1f47c7f8d9a3bf6a1e88b393f77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(3.78041934967041, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_df688af6921b18edb16207f474f74c21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df688af6921b18edb16207f474f74c21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15d9a7113bfa12bfd608353c9606c8bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(1988302.125, dtype='float32').reshape([]),
                paddle.to_tensor([0.33631986379623413], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_00d081309a056a4b79c4573efb67221c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(28358.865234375, dtype='float32').reshape([]),
                paddle.to_tensor([0.33631986379623413], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e580f8e3164fdf777d54f526aeb04b6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f73f8a5a9cc5d1ba89df5ec87c9b6c49
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.4161713123321533], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b9b83727703137a348ed81729f035070(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(35.01837158203125, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_783e70deb0192465ed587282498eb276(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(235.82322692871094, dtype='float32').reshape([]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cca3a89563bfaae2926fca7d6b4b3f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(136.27935791015625, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_059693cb7d0280a0a9c8616e8a2f9004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_059693cb7d0280a0a9c8616e8a2f9004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb83cf4564525eb540f14e838b36b197
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f72108d887a080c702f3b327c82d0ba4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(208722.890625, dtype='float32').reshape([]),
                paddle.to_tensor([0.17792560160160065], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7d258d0c075838fb66536b19fbdb12f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(237864.9375, dtype='float32').reshape([]),
                paddle.to_tensor([0.17792560160160065], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_ca0b7e81f9e03a15075fbad6f8c59dfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c905953f726f19f90c41ef4927424cef
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24608883261680603]]], dtype='float32').reshape([1, 1, 1]),
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


    class TestPrimitiveOp_06a3ce246e2f7c1f8c5faeb7af442700(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cdac9f6ef2c4003ed867abb8507f929
        def get_inputs(self):
            return [
                paddle.to_tensor([1110.6275634765625], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_f0d3141c00d0cea528930383dc4715b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca3dbde71d4cbf7e3bd6d9616fec7e12
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.0003679796354845166], [0.0011984009761363268], [0.0008312652935273945], [0.0037753786891698837], [0.01995982974767685], [0.00011072401684941724]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_eb5d5780cb9f0cbacd06088cde760f55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca3dbde71d4cbf7e3bd6d9616fec7e12
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.001364782452583313], [0.0004077347693964839], [0.030202772468328476], [0.0028741590213030577], [0.026446755975484848], [0.0006722644320689142]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_d63de0d2f16a25e850d4e762e2263620(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca3dbde71d4cbf7e3bd6d9616fec7e12
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.05796103551983833], [0.05304250493645668], [0.16478216648101807], [0.05082429572939873], [0.16708996891975403], [0.05151817202568054]]], dtype='float32').reshape([1, 6, 1]),
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


    class TestPrimitiveOp_1cbc81ed13921685f03c293c497277e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(9.806840896606445, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_648d2a63f046d77c151fd2fe1d557509(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(3.564072608947754, dtype='float32').reshape([]),
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


    
    class PrimitiveOp_3a4bc481bb5c7045cf305e9a5281f548(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1696, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1696, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_403f360d64a576271bb10504372076b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a4bc481bb5c7045cf305e9a5281f548
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_403f360d64a576271bb10504372076b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a4bc481bb5c7045cf305e9a5281f548
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb50bc3702e1b99f536c0018b2cc0f99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(-449226.28125, dtype='float32').reshape([]),
                paddle.to_tensor([0.49678489565849304], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_614bc489c4f6ed8058be48df312cd9ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(96541.46875, dtype='float32').reshape([]),
                paddle.to_tensor([0.49678489565849304], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_da53b59bf17a40f3561975403a32643f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(953.4589233398438, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_825da8829fcb5c7569f6743f4a591458(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68ded890f87b9d3fd6c362b4fcfc0e1b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.02909252792596817], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[-0.08513391762971878], [-0.0048834290355443954], [0.09082716703414917], [0.06779509782791138], [-0.02920367568731308], [0.020875904709100723], [0.09581932425498962], [-0.005090379621833563], [0.07913517206907272]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_6f9fdf4495a484c2cb2bcaa2b6b237c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68ded890f87b9d3fd6c362b4fcfc0e1b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.11926756799221039], [0.008719295263290405], [-0.030501816421747208], [-0.02372565120458603], [0.010544508695602417], [-0.05020035803318024], [0.0], [0.05388128384947777], [-0.07870669662952423]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.034133654087781906], [0.0038358664605766535], [0.06032535061240196], [0.04406944662332535], [-0.018659166991710663], [-0.029324453324079514], [0.09581932425498962], [0.04879090562462807], [0.0004284780006855726]], dtype='float32').reshape([9, 1]),
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


    class TestPrimitiveOp_c7553fccbb4d6c307a65bece6d151396(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fea08d98e460778978a09aef3cd730f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3117090165615082], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_73ec18632b408397d4c305a5a871dbeb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5517, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[5517, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4f50f2f6eb181a7bc7598aedf0982910(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73ec18632b408397d4c305a5a871dbeb
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f50f2f6eb181a7bc7598aedf0982910(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73ec18632b408397d4c305a5a871dbeb
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1ba1f4b7dfe626aa8260cb090dd0e20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(-207.77392578125, dtype='float32').reshape([]),
                paddle.to_tensor([0.10024213790893555], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_985656a6ccd2eb7d054a6a086ac111f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(3922.744140625, dtype='float32').reshape([]),
                paddle.to_tensor([0.10024213790893555], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_1b0d1ee4f60ca0fe8c2a66c87e443e91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_991840637b016f1ceba5f8049ac0c8ed
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.0, 0.0, -0.0, 0.0, 0.0, -0.0], dtype='float32').reshape([6]),
                paddle.to_tensor([0.07526008784770966, 0.02078755758702755, -0.02064768597483635, 0.003164077177643776, 0.009952809661626816, 0.030713750049471855], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_4f3ea69651256f3a5543d7cfb83ee148(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_991840637b016f1ceba5f8049ac0c8ed
        def get_inputs(self):
            return [
                paddle.to_tensor([0.07436105608940125, 0.038930028676986694, 0.004602316301316023, 0.08816459029912949, 0.04051024839282036, 0.03402898088097572], dtype='float32').reshape([6]),
                paddle.to_tensor([0.26359811425209045, 0.0007747879717499018, 0.06292464584112167, 0.14202694594860077, 0.04369882866740227, 0.15298713743686676], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_9828fc5d4d700f248d83deba1987eb94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_991840637b016f1ceba5f8049ac0c8ed
        def get_inputs(self):
            return [
                paddle.to_tensor([0.002785980701446533, -0.30777591466903687, 0.25031355023384094, 0.015211403369903564, -0.05733698606491089, 0.1896750032901764], dtype='float32').reshape([6]),
                paddle.to_tensor([0.20420202612876892, -0.0675412118434906, -0.08248728513717651, 0.20800691843032837, -0.1735844612121582, 0.16192829608917236], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_6d7d17cf2ac368419f7781faa940db76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_991840637b016f1ceba5f8049ac0c8ed
        def get_inputs(self):
            return [
                paddle.to_tensor([0.25374525785446167, -0.32804441452026367, -0.25219541788101196, -0.3383956551551819, -0.24549104273319244, -0.06895255297422409], dtype='float32').reshape([6]),
                paddle.to_tensor([0.29435500502586365, -0.17794977128505707, 0.016363710165023804, -0.21813932061195374, 0.1977015733718872, 0.17290432751178741], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_3c5c162e4073f70a837dc080d7a2ddf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_991840637b016f1ceba5f8049ac0c8ed
        def get_inputs(self):
            return [
                paddle.to_tensor([0.19734278321266174, 0.03200355917215347, 0.026051675900816917, 0.34692761301994324, 0.5951855778694153, 0.6268026828765869], dtype='float32').reshape([6]),
                paddle.to_tensor([1.1973427534103394, 1.0320035219192505, 1.026051640510559, 1.3469276428222656, 1.5951855182647705, 1.626802682876587], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_f6eec304184ec0aa2d152dd357fadb7e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1794, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1794, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_03287dcdf597970f8f3d5ebed2de2baf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6eec304184ec0aa2d152dd357fadb7e
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03287dcdf597970f8f3d5ebed2de2baf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6eec304184ec0aa2d152dd357fadb7e
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0024e648df23dd92ba80db589e9be3b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(211035.953125, dtype='float32').reshape([]),
                paddle.to_tensor([0.3157075345516205], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b291b5a1cf04fe53fe75f72c781d654b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(106355.296875, dtype='float32').reshape([]),
                paddle.to_tensor([0.3157075345516205], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9259adb0c3fee7fc7b1e9f3579c5b9a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(948.6329345703125, dtype='float32').reshape([]),
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


    
    class PrimitiveOp_9b9977d9971bda74c0d8c7f5a1a39379(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1504, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1504, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_30316a62de9f8dd7281f78e85ce99cd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b9977d9971bda74c0d8c7f5a1a39379
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30316a62de9f8dd7281f78e85ce99cd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b9977d9971bda74c0d8c7f5a1a39379
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e353bdb381cc4d3aa0613765d39345e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(28115.0546875, dtype='float32').reshape([]),
                paddle.to_tensor([0.28394192457199097], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9a5d01e68183dc2aeef5e8d748ec7e94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(85368.09375, dtype='float32').reshape([]),
                paddle.to_tensor([0.28394192457199097], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_9406fd7cb65e7e923032aafdc818be64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d2e452b8e4207992fdbf0ad3db96a44
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.2430521547794342], [0.24500004947185516]]], dtype='float32').reshape([1, 2, 1]),
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


    class TestPrimitiveOp_e0fcca84b731e5239a0ec5d2fd4242c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e074e766a0ef5ac615b8de21d9514981
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0809708684682846]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_aa6972d0d0afdf1d5a5dba128a67a12c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e074e766a0ef5ac615b8de21d9514981
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.006399311125278473]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.08737017959356308]], dtype='float32').reshape([1, 1]),
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


    class TestPrimitiveOp_81e8f594de65b23e1473c1b397c59c79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d618a1b4a2ab12bf286cb1ad270ff0e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-0.009171975776553154], [-0.0004534857871476561], [-0.13099893927574158], [-0.018913721665740013], [-0.02232457511126995], [0.0065928734838962555]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_59cd00b56911f9a793b05c9887a99ea4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d618a1b4a2ab12bf286cb1ad270ff0e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.026915011927485466], [0.03408973291516304], [0.1773102730512619], [0.013566361740231514], [0.0009194985032081604], [-0.011294983327388763]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.017743036150932312], [0.033636245876550674], [0.046311333775520325], [-0.0053473603911697865], [-0.02140507660806179], [-0.004702110309153795]], dtype='float32').reshape([6, 1]),
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


    class TestPrimitiveOp_9e846588dd8866568f3c3a735afac55f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7cb14f960aeab4808b86a45135ef35c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.2399139702320099]]], dtype='float32').reshape([1, 1, 1]),
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


    class TestPrimitiveOp_a53ec4c212f50a83f2f956c1a0b2ad7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(58.3995246887207, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5743dc51fd13b604e596f826383ed5c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(559.1926879882812, dtype='float32').reshape([]),
                paddle.to_tensor([4.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_190f3d75162485344c05f40ca0161b0f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2039, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2039, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_48ed3d153991b5392f046c0ae6f83c1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_190f3d75162485344c05f40ca0161b0f
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48ed3d153991b5392f046c0ae6f83c1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_190f3d75162485344c05f40ca0161b0f
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5004d5491440de6308438895c4f86a71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(66534.359375, dtype='float32').reshape([]),
                paddle.to_tensor([0.38607752323150635], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6962213fdde9757b0810b86e2bca03d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(116012.0546875, dtype='float32').reshape([]),
                paddle.to_tensor([0.38607752323150635], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_dae9fce7eb4eca6e69daa62501383230(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_812308383c8137982374a4e7c9413569
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.26358479261398315], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_a3fc1d3ee722bda55e08bcb8fb9d3ad4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4584, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4584, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_389b15e0249e224d26e597adee069152(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3fc1d3ee722bda55e08bcb8fb9d3ad4
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_389b15e0249e224d26e597adee069152(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3fc1d3ee722bda55e08bcb8fb9d3ad4
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e10502fe927e980fcd825eee4ae9184b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(125483.75, dtype='float32').reshape([]),
                paddle.to_tensor([0.004646401386708021], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9ee60cee9c51841e5c09bfb1934e30d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(260588.625, dtype='float32').reshape([]),
                paddle.to_tensor([0.004646401386708021], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_027d3e06d5616c33c197a4f8abba59cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cdac9f6ef2c4003ed867abb8507f929
        def get_inputs(self):
            return [
                paddle.to_tensor([316.174560546875], dtype='float32').reshape([1]),
                paddle.to_tensor(2434.0, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_2c0912b9cf4b8b9e79d4fb4440f02f60(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1071, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1071, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8a0ce365e64ad7d9e06c1523d76f56fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c0912b9cf4b8b9e79d4fb4440f02f60
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a0ce365e64ad7d9e06c1523d76f56fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c0912b9cf4b8b9e79d4fb4440f02f60
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58d9e135d57e5127e10a18edf0b57aa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(63740.50390625, dtype='float32').reshape([]),
                paddle.to_tensor([0.04321810603141785], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8656092a7a9e4fa1a4235b58ea829665(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(15261.7607421875, dtype='float32').reshape([]),
                paddle.to_tensor([0.04321810603141785], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_b03b1519832580936cc7adefeb768603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35e9add6de32eccdde61cac6037f317c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4150143563747406, 0.17286737263202667, 0.18879666924476624, 0.00761602446436882], [0.22656339406967163, 0.46918755769729614, 0.38365957140922546, 0.12311527878046036]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.30250710248947144, 0.2891940772533417, 0.37603989243507385, 0.36301806569099426], [0.42629826068878174, 0.40945640206336975, 0.46440133452415466, 0.19835348427295685]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_22c995b5d89f70d6edf8736f5cce22e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35e9add6de32eccdde61cac6037f317c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17972084879875183, 0.021701132878661156, 0.0599205382168293, 0.38248834013938904], [0.13160006701946259, 0.4700474739074707, 0.3898284435272217, 0.3776264488697052]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.45377030968666077, 0.47708168625831604, 0.3173177242279053, 0.03736255690455437], [0.439419686794281, 0.11878792941570282, 0.030491745099425316, 0.06279624998569489]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_b1f5cc14659d1efe1448246f607bd4aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0adff6a8ed62bb1e41203249d6ff0731
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.015536606311798096], [-0.028670985251665115], [0.08992450684309006], [-0.0018401051638647914], [0.0011996077373623848]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_151755b494aa00985c155b55586b499b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0adff6a8ed62bb1e41203249d6ff0731
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0403393991291523], [0.11921769380569458], [0.02897804230451584], [-0.00499636959284544], [0.0033686147071421146]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.024802792817354202], [0.09054671227931976], [0.1189025491476059], [-0.006836474873125553], [0.0045682224445044994]], dtype='float32').reshape([5, 1]),
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


    class TestPrimitiveOp_57968d4a7a8fc82b38ed65c6984a4ec4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1582d163c6fb0022e25492784192f798
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.2312423139810562], dtype='float32').reshape([1]),
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


    
    class PrimitiveOp_8a4c45b11cb612b807db375b69b5c6e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2370, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2370, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2d150f0e26ce2fb76290620e5f7c2389(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a4c45b11cb612b807db375b69b5c6e7
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d150f0e26ce2fb76290620e5f7c2389(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a4c45b11cb612b807db375b69b5c6e7
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f94259e3fbbc56d9348d3bcdd968e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(154219.03125, dtype='float32').reshape([]),
                paddle.to_tensor([0.0921860933303833], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ad22de7a43834e4047d46327d957a360(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(134749.84375, dtype='float32').reshape([]),
                paddle.to_tensor([0.0921860933303833], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_c4145ff6caf45a8b1d28191b1db47d6d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2993, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2993, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b9994f0c8eaed3e586b5d47c9d7afdb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4145ff6caf45a8b1d28191b1db47d6d
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9994f0c8eaed3e586b5d47c9d7afdb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4145ff6caf45a8b1d28191b1db47d6d
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a5d518f710335bd19dc213f9bdaa52e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(364645.03125, dtype='float32').reshape([]),
                paddle.to_tensor([0.4369841516017914], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5905110c4878c728f024b333a03dd20e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(170197.265625, dtype='float32').reshape([]),
                paddle.to_tensor([0.4369841516017914], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_db071e5fdf2507b9b6906a708f9e1dd6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3832, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3832, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_11a0061a8736cc44d3d16ec44ed26571(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db071e5fdf2507b9b6906a708f9e1dd6
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11a0061a8736cc44d3d16ec44ed26571(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db071e5fdf2507b9b6906a708f9e1dd6
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fda1ccf4564b0e78e98f1f953c15be0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(-1592628.125, dtype='float32').reshape([]),
                paddle.to_tensor([0.05049596726894379], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a0441e385f53f8820d003005dd0c7e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(217823.0, dtype='float32').reshape([]),
                paddle.to_tensor([0.05049596726894379], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_35e791ba48dfd8bc3710e05499953e44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_388cd211ba0040b84151d9f849c3e153
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.04804033041000366], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_05b98f5312b8417f26cab5e75650a7b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7cdbefa946b309a7d0147b82f7ccc9c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85e66ae86d14f3da044e60d13033fa23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(14.4016695022583, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_d7c9f10144931f1dfa1fb3fa2498e545(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17f1f74dd80dfa88292a6395ded48a7b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-0.06323745846748352], [-0.025388313457369804], [0.017503943294286728], [0.007693326100707054]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_b594e6d664520ccceaa54fabe2d3205f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17f1f74dd80dfa88292a6395ded48a7b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08949588239192963], [0.03676968067884445], [0.012986956164240837], [-0.008089913055300713]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.026258420199155807], [0.011381367221474648], [0.030490899458527565], [-0.0003965869836974889]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_a97f3757ed302a388c080161d8970b75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(3.78041934967041, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_8f29f012bd9a49f862fc8ee68c47c019(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fccd45b7b66929098585f489a9732d47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f29f012bd9a49f862fc8ee68c47c019
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fccd45b7b66929098585f489a9732d47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f29f012bd9a49f862fc8ee68c47c019
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be9979ce7ab7e59837b066ce16d10232(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(1988302.125, dtype='float32').reshape([]),
                paddle.to_tensor([0.33631986379623413], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ccc8eb7cbcce9c54b363c8ada8352316(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(28358.865234375, dtype='float32').reshape([]),
                paddle.to_tensor([0.33631986379623413], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_a66b75be03bca7f86d36a76908814450(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7c88a4b0e6c9053ad407f332e761931
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.4161713123321533], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5630437cd65debfd052f49bb1cba585d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(35.01837158203125, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_f3b61dd6870e127870f721d0e17de043(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(235.82322692871094, dtype='float32').reshape([]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4167e9d123a1cff2bf9b86588a0f3b1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(136.27935791015625, dtype='float32').reshape([]),
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


    
    class PrimitiveOp_00a23f31f225d425319e6b413d184d51(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 / input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4181, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4181, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_88488a9985919b4e56b1e0c0f01ac533(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00a23f31f225d425319e6b413d184d51
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88488a9985919b4e56b1e0c0f01ac533(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00a23f31f225d425319e6b413d184d51
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5df89b5260799c1ec06a5994d0eebb54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(208722.890625, dtype='float32').reshape([]),
                paddle.to_tensor([0.17792560160160065], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cac9d479048edf81ba17de148176c826(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb461ed1473e708df86d4965d867f485
        def get_inputs(self):
            return [
                paddle.to_tensor(237864.9375, dtype='float32').reshape([]),
                paddle.to_tensor([0.17792560160160065], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_8c7676d4d11179f5375b2c22c4e2cf98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24608883261680603]]], dtype='float32').reshape([1, 1, 1]),
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


    class TestPrimitiveOp_2c436f2187ef9e207e6a62ccb0d39405(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_151713f40442abd0afb75fc9e3ad549a
        def get_inputs(self):
            return [
                paddle.to_tensor([1110.6275634765625], dtype='float32').reshape([1]),
                paddle.to_tensor(8732.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_583d237ac57c4b7666406fffec30da37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.0003679796354845166], [0.0011984009761363268], [0.0008312652935273945], [0.0037753786891698837], [0.01995982974767685], [0.00011072401684941724]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_52c6d0fdb0d33bb278d4dbec58409848(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.001364782452583313], [0.0004077347693964839], [0.030202772468328476], [0.0028741590213030577], [0.026446755975484848], [0.0006722644320689142]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_da70c4d099516a48fa9216c1a1bb3563(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.05796103551983833], [0.05304250493645668], [0.16478216648101807], [0.05082429572939873], [0.16708996891975403], [0.05151817202568054]]], dtype='float32').reshape([1, 6, 1]),
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


    class TestPrimitiveOp_9ae83e98e566ab50d65d440a0060f1fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(9.806840896606445, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7efa70b7be76f3172370a367d323def1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(3.564072608947754, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_77162fccb23ad0378a76e0d57d423b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77162fccb23ad0378a76e0d57d423b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b51910e10a979753104572d7ad6c2777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-449226.28125, dtype='float32').reshape([]),
                paddle.to_tensor([0.49678489565849304], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_06c9f545b441c5cb19083c97be161b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(96541.46875, dtype='float32').reshape([]),
                paddle.to_tensor([0.49678489565849304], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5a202f8c366968da80dc3f590d798e9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(953.4589233398438, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_dca7ef59040627ce9a44168838ea470f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.02909252792596817], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[-0.08513391762971878], [-0.0048834290355443954], [0.09082716703414917], [0.06779509782791138], [-0.02920367568731308], [0.020875904709100723], [0.09581932425498962], [-0.005090379621833563], [0.07913517206907272]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_19f62d0c69e16998629901b2d1428b30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.11926756799221039], [0.008719295263290405], [-0.030501816421747208], [-0.02372565120458603], [0.010544508695602417], [-0.05020035803318024], [0.0], [0.05388128384947777], [-0.07870669662952423]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.034133654087781906], [0.0038358664605766535], [0.06032535061240196], [0.04406944662332535], [-0.018659166991710663], [-0.029324453324079514], [0.09581932425498962], [0.04879090562462807], [0.0004284780006855726]], dtype='float32').reshape([9, 1]),
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


    class TestPrimitiveOp_527ef7f285a54b46876b4509adda2618(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3027de34ed145e471b26ca7e9684a54
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3117090165615082], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0512a14f8be0115e5a7adeaa7b17575b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0512a14f8be0115e5a7adeaa7b17575b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea19c67bdccbce1d84dce2514d5a3a4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-207.77392578125, dtype='float32').reshape([]),
                paddle.to_tensor([0.10024213790893555], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_740b2fd44fd7877443aaa75c5f52d1c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(3922.744140625, dtype='float32').reshape([]),
                paddle.to_tensor([0.10024213790893555], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b02a61270ce5a82d16f8175b7789acab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.0, 0.0, -0.0, 0.0, 0.0, -0.0], dtype='float32').reshape([6]),
                paddle.to_tensor([0.07526008784770966, 0.02078755758702755, -0.02064768597483635, 0.003164077177643776, 0.009952809661626816, 0.030713750049471855], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_eaf4e1ee195261f4646cb6183770752b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.07436105608940125, 0.038930028676986694, 0.004602316301316023, 0.08816459029912949, 0.04051024839282036, 0.03402898088097572], dtype='float32').reshape([6]),
                paddle.to_tensor([0.26359811425209045, 0.0007747879717499018, 0.06292464584112167, 0.14202694594860077, 0.04369882866740227, 0.15298713743686676], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_6addad4c03917cd3f79f98d0f5fc280b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.002785980701446533, -0.30777591466903687, 0.25031355023384094, 0.015211403369903564, -0.05733698606491089, 0.1896750032901764], dtype='float32').reshape([6]),
                paddle.to_tensor([0.20420202612876892, -0.0675412118434906, -0.08248728513717651, 0.20800691843032837, -0.1735844612121582, 0.16192829608917236], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_d4d9a01ff5df415fed9490836fd9d705(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.25374525785446167, -0.32804441452026367, -0.25219541788101196, -0.3383956551551819, -0.24549104273319244, -0.06895255297422409], dtype='float32').reshape([6]),
                paddle.to_tensor([0.29435500502586365, -0.17794977128505707, 0.016363710165023804, -0.21813932061195374, 0.1977015733718872, 0.17290432751178741], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_46e6d18fb34285df320930f1e474b8dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c61b24e5e854c926c41c52f551490d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.19734278321266174, 0.03200355917215347, 0.026051675900816917, 0.34692761301994324, 0.5951855778694153, 0.6268026828765869], dtype='float32').reshape([6]),
                paddle.to_tensor([1.1973427534103394, 1.0320035219192505, 1.026051640510559, 1.3469276428222656, 1.5951855182647705, 1.626802682876587], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_30d78046c0a2dad6f6384d2844182845(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30d78046c0a2dad6f6384d2844182845(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75e5cf69da0bbf9728ce2ab839624610(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(211035.953125, dtype='float32').reshape([]),
                paddle.to_tensor([0.3157075345516205], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_732331af0ac8476bfaf654f4bd13e3bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(106355.296875, dtype='float32').reshape([]),
                paddle.to_tensor([0.3157075345516205], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c95c52c0d7dd4f045b003236df817652(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(948.6329345703125, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_beccc24684e03ece6d43aea8d3950f6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beccc24684e03ece6d43aea8d3950f6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea64e8804335e022f58943bde2bba9f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(28115.0546875, dtype='float32').reshape([]),
                paddle.to_tensor([0.28394192457199097], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7e7da7232e278bdbf31b08687fc86f03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(85368.09375, dtype='float32').reshape([]),
                paddle.to_tensor([0.28394192457199097], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_47aa69004b5184169bcebe7b9069f36f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.2430521547794342], [0.24500004947185516]]], dtype='float32').reshape([1, 2, 1]),
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


    class TestPrimitiveOp_51add1ce07ba6a789eb220bded02f00f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0809708684682846]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_d8c1b37093abae1681a7866e83e099bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.006399311125278473]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.08737017959356308]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_6b3131d3f47f5d5832a962476218308f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-0.009171975776553154], [-0.0004534857871476561], [-0.13099893927574158], [-0.018913721665740013], [-0.02232457511126995], [0.0065928734838962555]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_1377573c5cc54d8a1428a5a9654ca85d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.026915011927485466], [0.03408973291516304], [0.1773102730512619], [0.013566361740231514], [0.0009194985032081604], [-0.011294983327388763]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.017743036150932312], [0.033636245876550674], [0.046311333775520325], [-0.0053473603911697865], [-0.02140507660806179], [-0.004702110309153795]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_9cba196b3b2a5d270f93614750d7469b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60af33618c491dd5c3b9b2e81a6b0f27
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.2399139702320099]]], dtype='float32').reshape([1, 1, 1]),
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


    class TestPrimitiveOp_52c0ae16588325a46e71e70206df662a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(58.3995246887207, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ac59ec19e81ad2ebc687900b9075ed57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(559.1926879882812, dtype='float32').reshape([]),
                paddle.to_tensor([4.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e3f5e8668dd04ab8a3a114e85d1932c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3f5e8668dd04ab8a3a114e85d1932c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef6d7a260ea383091c139875e2b48020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(66534.359375, dtype='float32').reshape([]),
                paddle.to_tensor([0.38607752323150635], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6fd18195dd7997aed5375c6f0fb2b80a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(116012.0546875, dtype='float32').reshape([]),
                paddle.to_tensor([0.38607752323150635], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_2bb245a3b8137ee19cc889c747e591d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3027de34ed145e471b26ca7e9684a54
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.26358479261398315], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1a9afc0a63b2124ee20e625ddcba4311(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a9afc0a63b2124ee20e625ddcba4311(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20a7df03dc523325dc2ea8a956188c9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(125483.75, dtype='float32').reshape([]),
                paddle.to_tensor([0.004646401386708021], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a9b23b48638404f356c2e6493dd4f1c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(260588.625, dtype='float32').reshape([]),
                paddle.to_tensor([0.004646401386708021], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_73b32a76a318e2a3dfc84b227c557055(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_151713f40442abd0afb75fc9e3ad549a
        def get_inputs(self):
            return [
                paddle.to_tensor([316.174560546875], dtype='float32').reshape([1]),
                paddle.to_tensor(2434.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_e708a11e7f3e7eeb4786322be47b6165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e708a11e7f3e7eeb4786322be47b6165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0edc65dfdc6d0b015aa9c308337ba33d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(63740.50390625, dtype='float32').reshape([]),
                paddle.to_tensor([0.04321810603141785], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9048c9091e2da1ee790edd4496f37842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(15261.7607421875, dtype='float32').reshape([]),
                paddle.to_tensor([0.04321810603141785], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_fff6fe2903691518dbb3572e1e5db7a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4150143563747406, 0.17286737263202667, 0.18879666924476624, 0.00761602446436882], [0.22656339406967163, 0.46918755769729614, 0.38365957140922546, 0.12311527878046036]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.30250710248947144, 0.2891940772533417, 0.37603989243507385, 0.36301806569099426], [0.42629826068878174, 0.40945640206336975, 0.46440133452415466, 0.19835348427295685]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_8c1e277c8fc9095b4d422a75607c0b19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17972084879875183, 0.021701132878661156, 0.0599205382168293, 0.38248834013938904], [0.13160006701946259, 0.4700474739074707, 0.3898284435272217, 0.3776264488697052]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([[0.45377030968666077, 0.47708168625831604, 0.3173177242279053, 0.03736255690455437], [0.439419686794281, 0.11878792941570282, 0.030491745099425316, 0.06279624998569489]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_7c1e390d66a3a02e238a8284cac699c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.015536606311798096], [-0.028670985251665115], [0.08992450684309006], [-0.0018401051638647914], [0.0011996077373623848]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_01a8227f6b620a42b64da3fa1734c42f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0403393991291523], [0.11921769380569458], [0.02897804230451584], [-0.00499636959284544], [0.0033686147071421146]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.024802792817354202], [0.09054671227931976], [0.1189025491476059], [-0.006836474873125553], [0.0045682224445044994]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_f215cd39b540f793dd661c0b88e91558(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3027de34ed145e471b26ca7e9684a54
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.2312423139810562], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_04bc64c02879712d06d2a6873e0abf44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04bc64c02879712d06d2a6873e0abf44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4dae842277328be20948c2ae90695e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(154219.03125, dtype='float32').reshape([]),
                paddle.to_tensor([0.0921860933303833], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_75736d9b71fa1ce2187b83b41e602ad9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(134749.84375, dtype='float32').reshape([]),
                paddle.to_tensor([0.0921860933303833], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c0091f4c8a3d3d87ede0fbcdd7c183d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0091f4c8a3d3d87ede0fbcdd7c183d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96e0049ddc974702c8d20e8e92d1c6c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(364645.03125, dtype='float32').reshape([]),
                paddle.to_tensor([0.4369841516017914], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9485bad0b7d48672823e6e02ba9d0974(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(170197.265625, dtype='float32').reshape([]),
                paddle.to_tensor([0.4369841516017914], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cb0deab92d4678cbb081e896b8fd699c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb0deab92d4678cbb081e896b8fd699c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3369327041e6ff9f6fc42ab0de5be200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(-1592628.125, dtype='float32').reshape([]),
                paddle.to_tensor([0.05049596726894379], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_14d3ea0960d9e3fa8e0bd1767e78ac00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(217823.0, dtype='float32').reshape([]),
                paddle.to_tensor([0.05049596726894379], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ce0fcd77ac45937d2cccdfc63c2281f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3027de34ed145e471b26ca7e9684a54
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.04804033041000366], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9f17c4071a87bab74c185606e898b17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68e984b4e450142c6e836f7b5b620ec6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_724fecb829d8f33326a9cb3444f219b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(14.4016695022583, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_91bcbe803f31bf03ddbaca2c7a0ac7ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-0.06323745846748352], [-0.025388313457369804], [0.017503943294286728], [0.007693326100707054]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_4c06d175098e85308eabcf7194949cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08949588239192963], [0.03676968067884445], [0.012986956164240837], [-0.008089913055300713]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.026258420199155807], [0.011381367221474648], [0.030490899458527565], [-0.0003965869836974889]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_a600e1f47c7f8d9a3bf6a1e88b393f77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(3.78041934967041, dtype='float32').reshape([]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_62d3546b686a0e693790d2eab362fe80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62d3546b686a0e693790d2eab362fe80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15d9a7113bfa12bfd608353c9606c8bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(1988302.125, dtype='float32').reshape([]),
                paddle.to_tensor([0.33631986379623413], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_00d081309a056a4b79c4573efb67221c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(28358.865234375, dtype='float32').reshape([]),
                paddle.to_tensor([0.33631986379623413], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c2b727f3e876f391f38d8df8266fb4aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3027de34ed145e471b26ca7e9684a54
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.4161713123321533], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b9b83727703137a348ed81729f035070(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(35.01837158203125, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_783e70deb0192465ed587282498eb276(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(235.82322692871094, dtype='float32').reshape([]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cca3a89563bfaae2926fca7d6b4b3f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(136.27935791015625, dtype='float32').reshape([]),
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


    class TestPrimitiveOp_550d9f0219b02c6a940b2a01e17f4475(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_550d9f0219b02c6a940b2a01e17f4475(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3975ce6f7105f8ff235ff0f3114a6e
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f72108d887a080c702f3b327c82d0ba4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(208722.890625, dtype='float32').reshape([]),
                paddle.to_tensor([0.17792560160160065], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7d258d0c075838fb66536b19fbdb12f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63daf8ae3628379c7b72f50d6eec09fc
        def get_inputs(self):
            return [
                paddle.to_tensor(237864.9375, dtype='float32').reshape([]),
                paddle.to_tensor([0.17792560160160065], dtype='float32').reshape([1]),
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